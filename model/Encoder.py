from torch import nn
import torchvision as tv
from .ASPP import ASPP
from .AtrousResNet import resnet50


class Encoder(nn.Module):

    def __init__(self, BatchNorm):
        super(Encoder, self).__init__()

        self.resnet50 = resnet50(BatchNorm)
        self.aspp = ASPP(2048, 256, BatchNorm)

        # self._initialize_weights()

    def _initialize_weights(self):

        pretrained_resnet50 = tv.models.resnet50(pretrained=True)
        pretrained_dict = pretrained_resnet50.state_dict()

        atrous_resnet_dict = self.resnet50.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in atrous_resnet_dict}

        atrous_resnet_dict.update(pretrained_dict)

        self.resnet50.load_state_dict(atrous_resnet_dict)

    def forward(self, x):

        x, skip_connection1, skip_connection2, skip_connection3, max_index = self.resnet50(x)

        # x = self.conv(x)
        x = self.aspp(x)

        return x, skip_connection1, skip_connection2, skip_connection3, max_index

