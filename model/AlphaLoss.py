import torch as t
from torch import nn


class AlphaLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(AlphaLoss, self).__init__()
        self.eps = eps

    def forward(self, predict, truth, size_average=True):

        diff = predict - truth
        losses = t.sqrt(diff.pow(2) + self.eps*self.eps)

        return losses.mean() if size_average else losses.sum()
