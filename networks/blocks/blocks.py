# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/1/12
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers



class ChannelPool(nn.Module):
    def __init__(self, types):
        super(ChannelPool, self).__init__()
        if types == 'avg':
            self.poolingx = nn.AdaptiveAvgPool1d(1)
        elif types == 'max':
            self.poolingx = nn.AdaptiveMaxPool1d(1)
        else:
            raise Exception('inner error')

    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = self.poolingx(input)  # b,w*h,c ->  b,w*h,1
        _, _, c = pooled.size()
        return pooled.view(n, c, w, h)




