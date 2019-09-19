from torch import nn
import torchvision as tv
from .ASPP import ASPP
from .AtrousResNet import resnet50


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.resnet50 = resnet50()
        self.aspp = ASPP(in_channels=2048, out_channels=256)

        self._initialize_weights()

    def _initialize_weights(self):
        # init atrous_resnet50 with the pretrained resnet
        pretrained_resnet50 = tv.models.resnet50(pretrained=True)
        pretrained_dict = pretrained_resnet50.state_dict()

        atrous_resnet_dict = self.resnet50.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in atrous_resnet_dict}

        atrous_resnet_dict.update(pretrained_dict)

        self.resnet50.load_state_dict(atrous_resnet_dict)
        # init aspp
        for m in self.aspp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x, skip_connection1, skip_connection2, skip_connection3, max_index = self.resnet50(x)

        x = self.aspp(x)

        return x, skip_connection1, skip_connection2, skip_connection3, max_index

