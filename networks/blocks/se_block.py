# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/1/12
"""
From:https://arxiv.org/pdf/1709.01507.pdf
code:https://github.com/moskomule/senet.pytorch
"""
from torch import nn


class SEBlock(nn.Module):
    """docstring for SEBlock"""

    def __init__(self, channel, reducation=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reducation),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reducation, channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, w, h = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y = self.fc(y1).view(b, c, 1, 1)
        return x * y