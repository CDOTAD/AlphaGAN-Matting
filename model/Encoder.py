from torch import nn
import torchvision as tv
from .ASPP import ASPP
from .AtrousResNet import resnet50


class Encoder(nn.Module):

    def __init__(self, BatchNorm):
        super(Encoder, self).__init__()

        self.resnet50 = resnet50(BatchNorm)
        self.aspp = ASPP(2048, 256, BatchNorm)

    def forward(self, x):

        x, skip_connection1, skip_connection2, skip_connection3, max_index = self.resnet50(x)

        # x = self.conv(x)
        x = self.aspp(x)

        return x, skip_connection1, skip_connection2, skip_connection3, max_index

