# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/1/12
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.components.gaussian_smoothing import GaussianSmoothing


class BasicLearningBlock(nn.Module):
    """docstring for BasicLearningBlock"""

    def __init__(self, channel):
        super(BasicLearningBlock, self).__init__()
        self.rconv1 = nn.Conv2d(channel, channel * 2, 3, padding=1, bias=False)
        self.rbn1 = nn.BatchNorm2d(channel * 2)
        self.rconv2 = nn.Conv2d(channel * 2, channel, 3, padding=1, bias=False)
        self.rbn2 = nn.BatchNorm2d(channel)

    def forward(self, feature):
        return F.elu(self.rbn2(self.rconv2(F.elu(self.rbn1(self.rconv1(feature))))))


class GlobalAttentionModule(nn.Module):
    """docstring for GlobalAttentionModule"""

    def __init__(self, channel, reducation=16):
        super(GlobalAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel * 2, channel // reducation),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reducation, channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, w, h = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y2 = self.max_pool(x).view(b, c)
        y = self.fc(torch.cat([y1, y2], 1)).view(b, c, 1, 1)
        return x * y


class SpatialAttentionModule(nn.Module):
    """docstring for SpatialAttentionModule"""

    def __init__(self, channel, reducation=16):
        super(SpatialAttentionModule, self).__init__()
        self.avg_pool = ChannelPool('avg')
        self.max_pool = ChannelPool('max')
        self.fc = nn.Sequential(
            nn.Conv2d(2, reducation, 7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(reducation, 1, 7, stride=1, padding=3),
            nn.Sigmoid())

    def forward(self, x):
        b, c, w, h = x.size()
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        y = self.fc(torch.cat([y1, y2], 1))
        yr = 1 - y
        return y, yr


class GlobalAttentionModuleJustSigmoid(nn.Module):
    """docstring for GlobalAttentionModule"""

    def __init__(self, channel, reducation=16):
        super(GlobalAttentionModuleJustSigmoid, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel * 2, channel // reducation),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reducation, channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, w, h = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y2 = self.max_pool(x).view(b, c)
        y = self.fc(torch.cat([y1, y2], 1)).view(b, c, 1, 1)
        return y


class RASC(nn.Module):
    def __init__(self, channel, type_of_connection=BasicLearningBlock):
        super(RASC, self).__init__()
        self.connection = type_of_connection(channel)
        self.background_attention = GlobalAttentionModule(channel, 16)
        self.mixed_attention = GlobalAttentionModule(channel, 16)
        self.spliced_attention = GlobalAttentionModule(channel, 16)
        self.gaussianMask = GaussianSmoothing(1, 5, 1)

    def forward(self, feature, mask):
        _, _, w, _ = feature.size()
        _, _, mw, _ = mask.size()
        # binaryfiy
        # selected the feature from the background as the additional feature to masked splicing feature.
        if w != mw:
            mask = torch.round(F.avg_pool2d(mask, 2, stride=mw // w))
        reverse_mask = -1 * (mask - 1)
        # here we add gaussin filter to mask and reverse_mask for better harimoization of edges.

        mask = self.gaussianMask(F.pad(mask, (2, 2, 2, 2), mode='reflect'))
        reverse_mask = self.gaussianMask(F.pad(reverse_mask, (2, 2, 2, 2), mode='reflect'))

        background = self.background_attention(feature) * reverse_mask
        selected_feature = self.mixed_attention(feature)
        spliced_feature = self.spliced_attention(feature)
        spliced = (self.connection(spliced_feature) + selected_feature) * mask
        return background + spliced
