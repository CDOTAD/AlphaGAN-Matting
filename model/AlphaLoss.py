import torch as t
from torch import nn


class AlphaLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(AlphaLoss, self).__init__()
        self.eps = eps

    def forward(self, predict, truth, trimap):

        wi = t.zeros(trimap.shape)
        wi[((trimap*0.5 + 0.5)*255) == 128] = 1.
        t_wi = wi.cuda()
        unknown_region_size = t_wi.sum()

        diff = predict - truth
        losses = t.sqrt(diff.pow(2) + self.eps*self.eps)
        loss = (losses * t_wi).sum() / (unknown_region_size + 1.)

        return loss
