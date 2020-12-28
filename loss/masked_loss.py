# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/25
import torch
from torch.nn.modules.loss import L1Loss, MSELoss


class MaskedL1Loss(L1Loss):
    def __init__(self, lamb=0.1, size_average=None, reduce=None, reduction='mean'):
        super(MaskedL1Loss, self).__init__(size_average, reduce, reduction)
        self.lamb = lamb

    def forward(self, input, target, mask):
        pass


class MaskedMSELoss(MSELoss):
    def __init__(self, lamb=0.1, size_average=None, reduce=None, reduction='mean'):
        super(MaskedL1Loss, self).__init__(size_average, reduce, reduction)
        self.lamb = lamb

    def forward(self, input, target, mask):
        pass
