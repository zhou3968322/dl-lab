# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/25
import torch
from torch.nn.modules.loss import L1Loss, MSELoss


class MaskedL1Loss(L1Loss):
    def __init__(self, lamb=0.1, size_average=None, reduce=None, reduction='mean'):
        super(MaskedL1Loss, self).__init__(size_average, reduce, reduction)
        self.lamb = lamb

    def forward(self, pred, target, mask=None):
        # mask 为None的时候target is mask
        b, c, h, w = pred.size()
        l1_diff = torch.abs(target - pred)
        if mask is not None:
            hard = mask == 0.0
            easy = mask == 1.0
        else:
            hard = target == 0.0
            easy = target == 1.0
        loss = self.lamb * (torch.sum(l1_diff * easy) / (h * w)) + (1 - self.lamb) * (torch.sum(l1_diff * hard) / (h * w))
        return loss


class MaskedMSELoss(MSELoss):
    def __init__(self, lamb=0.1, size_average=None, reduce=None, reduction='mean'):
        super(MaskedL1Loss, self).__init__(size_average, reduce, reduction)
        self.lamb = lamb

    def forward(self, input, target, mask):
        pass

