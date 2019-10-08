from torch import nn
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


# atrous_ResNet
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, BatchNorm, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        if dilation != 1:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                   padding=dilation, dilation=dilation, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, BatchNorm, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv_1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.layer1 = self._make_layer(block, 64, layers[0], BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], BatchNorm, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], BatchNorm, stride=2, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], BatchNorm, stride=2, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, BatchNorm, stride=1, dilation=1):
        downsample = None
        downsample_stride = stride if dilation == 1 else 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=downsample_stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, BatchNorm, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, x):
        skip_connection1 = x  # 320 x 320

        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print('resnet50.forward.x.size : ', x.size())
        skip_connection2 = x  # 160 x 160

        x, max_index = self.maxpool(x)

        x = self.layer1(x)
        # print('resnet50.forward.x.size : ', x.size())
        skip_connection3 = x  # 80 x 80

        x = self.layer2(x)
        # print('resnet50.forward.x.size : ', x.size())
        x = self.layer3(x)
        # print('resnet50.forward.x.size : ', x.size())
        x = self.layer4(x)
        # print('resnet50.forward.x.size : ', x.size())

        return x, skip_connection1, skip_connection2, skip_connection3, max_index


def resnet50(BatchNorm):

    model = ResNet(Bottleneck, [3, 4, 6, 3], BatchNorm)

    return model
