import torch as t
from torch import nn
import torch.nn.functional as F
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

'''
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
'''


class BilinearUpSample(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BilinearUpSample, self).__init__()

        '''
        self.conv = nn.Sequential(
            conv3x3(in_planes=in_planes, out_planes=out_planes),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        '''
    def forward(self, x):
        in_size = x.size()
        n, c, h, w = in_size
        x = F.interpolate(x, size=(h*2, w*2), mode='bilinear', align_corners=True)
        return x


# decoder
class Decoder(nn.Module):

    def __init__(self, BatchNorm):
        super(Decoder, self).__init__()

        # bilinear
        self.bilinear_1 = BilinearUpSample(in_planes=256, out_planes=256)
        # output: 256 x 80 x 80

        self.skip_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=48, kernel_size=1, bias=False),
            BatchNorm(48),
            nn.ReLU(inplace=True)

        )

        # deconv1_x
        self.deconv1_x = nn.Sequential(
            nn.Conv2d(in_channels=256+48, out_channels=256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False),
            BatchNorm(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
            BatchNorm(64),
            nn.ReLU(inplace=True)
        )
        # output: 64 x 80 x 80

        # unpooling
        self.unpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # output: 64 x 160 x 160

        self.skip_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, bias=False),
            BatchNorm(32),
            nn.ReLU(inplace=True)
        )

        # deconv2_x
        self.deconv2_x = nn.Sequential(
            nn.Conv2d(in_channels=64+32, out_channels=64, kernel_size=3, padding=1, bias=False),
            BatchNorm(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            BatchNorm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            BatchNorm(32),
            nn.ReLU(inplace=True)
        )
        # output: 32 x 320 x 320

        # deconv3_x
        self.deconv3_x = nn.Sequential(
            nn.Conv2d(in_channels=32+3, out_channels=32, kernel_size=3, padding=1, bias=False),
            BatchNorm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
            BatchNorm(32),
            nn.ReLU(inplace=True)
        )
        # output: 32 x 320 x 320

        # deconv4_x
        self.deconv4_x = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        # output: 1 x 320 x 320

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x, skip_connection1, skip_connection2, skip_connection3, max_index = x
        x = self.bilinear_1(x)

        skip_connection3 = self.skip_3(skip_connection3)

        x = t.cat([x, skip_connection3], dim=1)
        x = self.deconv1_x(x)

        x = self.unpooling(x, max_index)
        skip_connection2 = self.skip_2(skip_connection2)
        x = t.cat([x, skip_connection2], dim=1)
        x = self.deconv2_x(x)

        skip_connection1 = skip_connection1[:, 0:3, :, :]
        x = t.cat([x, skip_connection1], dim=1)
        x = self.deconv3_x(x)
        x = self.deconv4_x(x)

        return x