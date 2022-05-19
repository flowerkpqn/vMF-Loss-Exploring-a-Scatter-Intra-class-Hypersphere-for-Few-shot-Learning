#-*- coding=utf-8 -*-
#@Time:4/4/2022 下午 7:06
#@Author:kpqn
#@File:loss.py
#@software:PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosLoss(nn.Linear):
    r"""
    Cosine Loss
    """

    def __init__(self, in_features, out_features, bias=False):
        super(CosLoss, self).__init__(in_features, out_features, bias)
        self.s_ = torch.nn.Parameter(torch.zeros(1))

    def loss(self, Z, target):
        s = F.softplus(self.s_).add(1.)
        l = F.cross_entropy(Z*s, target, weight=None, ignore_index=-100, reduction='mean')
        return l

    def forward(self,query, shot):
        logit = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1),
                         self.bias)  # [N x out_features]
        return logit


class kCosLoss(CosLoss):
    r"""
    t-vMF Loss
    """

    def __init__(self, in_features, out_features, bias=False, kappa=16):
        super(kCosLoss, self).__init__(in_features, out_features, bias)
        self.register_buffer('kappa', torch.Tensor([kappa]))

    def forward(self, query, shot):
        a=F.normalize(query, p=2, dim=1)
        b=F.normalize(shot, p=2, dim=1)
        cosine = F.linear(a, b,
                          None)  # [N x out_features]
        # logit = (1. + cosine).div(1. + (1. - cosine).mul(self.kappa)) - 1.
        logit = torch.div(2*(torch.exp(self.kappa*cosine)-torch.exp(-self.kappa)),torch.exp(self.kappa)-torch.exp(-self.kappa)) - 1.
        if self.bias is not None:
            logit+=self.bias

        return logit

    def extra_repr(self):
        return super(kCosLoss, self).extra_repr() + ', kappa={}'.format(self.kappa)
