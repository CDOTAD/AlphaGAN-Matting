import torch as t
from torch import nn
import torch.nn.functional as F


def depth_sep_dilated_conv_3x3_bn(in_channels, out_channels, padding, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                  groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


class _AsppBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super(_AsppBlock, self).__init__()

        if kernel_size == 1:
            self.atrous_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        else:
            self.atrous_conv = depth_sep_dilated_conv_3x3_bn(in_channels, out_channels, dilation_rate, dilation_rate)
        '''
        if dilation_rate == 1:
            self.atrous_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, bias=False)
        else:
            self.atrous_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                         dilation=dilation_rate, padding=dilation_rate, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        '''
    def forward(self, _input):
        x = self.atrous_conv(_input)
        return x


# input batch x 2048 x 40 x 40
class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        self.aspp_1 = _AsppBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, dilation_rate=1)
        self.aspp_6 = _AsppBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation_rate=6)
        self.aspp_12 = _AsppBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation_rate=12)
        self.aspp_18 = _AsppBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation_rate=18)

        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=5 * out_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(True)
        )

    def forward(self, x):
        aspp1 = self.aspp_1(x)  # 256
        aspp6 = self.aspp_6(x)  # 256
        aspp12 = self.aspp_12(x)  # 256
        aspp18 = self.aspp_18(x)  # 256

        im_p = self.image_pooling(x)  # 256
        im_p = F.interpolate(im_p, size=aspp18.size()[2:], mode='nearest')
        aspp = [aspp1, aspp6, aspp12, aspp18, im_p]
        aspp = t.cat(aspp, dim=1)

        return self.conv1(aspp)
