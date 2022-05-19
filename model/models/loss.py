#-*- coding=utf-8 -*-
#@Time:4/4/2022 下午 7:06
#@Author:kpqn
#@File:loss.py
#@software:PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F


class kCosLogit(nn.Linear):
    r"""
    t-vMF Loss
    """

    def __init__(self, kappa, in_features, out_features, bias=False, ):
        super(kCosLogit, self).__init__(in_features, out_features, bias)
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
        return super(kCosLogit, self).extra_repr() + ', kappa={}'.format(self.kappa)
