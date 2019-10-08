import torch as t
from torch import nn


class AlphaLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(AlphaLoss, self).__init__()
        self.eps = eps

    def forward(self, predict, truth, unknown_region_size):

        diff = (predict - truth)
        losses = t.sqrt(diff.pow(2) + self.eps*self.eps)
        loss = losses.sum() / (unknown_region_size + 1.)

        return loss
