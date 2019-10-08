import torch as t
from torch import nn
import torch.nn.functional as F
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class _AsppBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate, BatchNorm):
        super(_AsppBlock, self).__init__()
        if dilation_rate == 1:
            self.atrous_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, bias=False)
        else:
            self.atrous_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                         dilation=dilation_rate, padding=dilation_rate, bias=False)

        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, _input):
        x = self.atrous_conv(_input)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# input batch x 2048 x 40 x 40
class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels, BatchNorm):
        super(ASPP, self).__init__()

        self.aspp_1 = _AsppBlock(in_channels, 256, kernel_size=1, dilation_rate=1, BatchNorm=BatchNorm)
        self.aspp_6 = _AsppBlock(in_channels, 256, kernel_size=3, dilation_rate=6, BatchNorm=BatchNorm)
        self.aspp_12 = _AsppBlock(in_channels, 256, kernel_size=3, dilation_rate=12, BatchNorm=BatchNorm)
        self.aspp_18 = _AsppBlock(in_channels, 256, kernel_size=3, dilation_rate=18, BatchNorm=BatchNorm)

        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=5 * out_channels, out_channels=out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(True)
        )

        self._init_weight()

    def forward(self, x):
        aspp1 = self.aspp_1(x)  # 256
        aspp6 = self.aspp_6(x)  # 256
        aspp12 = self.aspp_12(x)  # 256
        aspp18 = self.aspp_18(x)  # 256

        im_p = self.image_pooling(x)  # 256
        im_p = F.interpolate(im_p, size=aspp18.size()[2:], mode='bilinear', align_corners=True)
        aspp = [aspp1, aspp6, aspp12, aspp18, im_p]
        aspp = t.cat(aspp, dim=1)

        return self.conv1(aspp)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
