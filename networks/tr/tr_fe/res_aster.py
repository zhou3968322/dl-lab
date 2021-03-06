# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/2/1
import torch
import torch.nn as nn
import torchvision

import sys
import math


# from config import get_args
# global_args = get_args(sys.argv[1:])


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def get_sinusoid_encoding(n_position, feat_dim, wave_length=10000):
    # [n_position]
    positions = torch.arange(0, n_position)  # .cuda()
    # [feat_dim]
    dim_range = torch.arange(0, feat_dim)  # .cuda()
    dim_range = torch.pow(wave_length, 2 * (dim_range // 2) / feat_dim)
    # [n_position, feat_dim]
    angles = positions.unsqueeze(1) / dim_range.unsqueeze(0)
    angles = angles.float()
    angles[:, 0::2] = torch.sin(angles[:, 0::2])
    angles[:, 1::2] = torch.cos(angles[:, 1::2])
    return angles


class AsterBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AsterBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNetAster(nn.Module):
    """For aster or crnn
       borrowed from: https://github.com/ayumiymk/aster.pytorch
    """

    def __init__(self, input_channel=1, output_channel=512, n_group=1):
        super(ResNetAster, self).__init__()
        self.n_group = n_group

        input_channel = input_channel
        self.layer0 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.inplanes = 32
        self.layer1 = self._make_layer(32, 3, [2, 2])  # [16, 50]
        self.layer2 = self._make_layer(64, 4, [2, 2])  # [8, 25]
        self.layer3 = self._make_layer(128, 6, [2, 2])  # [4, 25]
        self.layer4 = self._make_layer(256, 6, [1, ])  # [2, 25]
        self.layer5 = self._make_layer(output_channel, 3, [1, 1])  # [1, 25]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != [1, 1] or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(AsterBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(AsterBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)  # x5 is visual features

        b, c, h, w = x5.shape
        visual_feature = x5.permute(0, 1, 3, 2)
        visual_feature = visual_feature.contiguous().view(b, c, -1)
        visual_feature = visual_feature.permute(0, 2, 1)  # batch, seq, feature
        return visual_feature


def numel(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    x = torch.randn(3, 1, 64, 256)
    net = ResNetAster()
    encoder_feat = net(x)
    print(encoder_feat.size())  # 3*512*h/4*w/4

    num_params = numel(net)
    print(f'Number of parameters: {num_params}')
