# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/2/1
import torch
from torch import nn
from torchvision.models.resnet import resnet50, resnet101, resnet152


class ResNetFPN(nn.Module):
    def __init__(self, num_layers=50):
        super(ResNetFPN, self).__init__()
        self._num_layers = num_layers
        self._layers = {}

        self._init_head_tail()
        self.out_planes = self.fpn.planes

    def forward(self, x):
        c2 = self.head1(x)
        c3 = self.head2(c2)
        c4 = self.head3(c3)
        c5 = self.head4(c4)
        p3, p4, p5 = self.fpn(c3, c4, c5)
        # net_conv = [p2, p3, p4, p5]

        # return p2, [x, self.resnet.conv1(x), c2]
        # use p3 as features
        b, c, h, w = p3.shape
        visual_feature = p3.permute(0, 1, 3, 2)
        visual_feature = visual_feature.contiguous().view(b, c, -1)
        visual_feature = visual_feature.permute(0, 2, 1)  # batch, seq, feature
        return visual_feature

    def _init_head_tail(self):
        # choose different blocks for different number of layers
        if self._num_layers == 50:
            self.resnet = resnet50()

        elif self._num_layers == 101:
            self.resnet = resnet101()

        elif self._num_layers == 152:
            self.resnet = resnet152()

        else:
            # other numbers are not supported
            raise NotImplementedError

        # Build Building Block for FPN
        self.fpn = BuildBlock()
        self.head1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1)  # /4
        self.head2 = nn.Sequential(self.resnet.layer2)  # /8
        self.head3 = nn.Sequential(self.resnet.layer3)  # /16
        self.head4 = nn.Sequential(self.resnet.layer4)  # /32


class BuildBlock(nn.Module):
    def __init__(self, planes=512):
        super(BuildBlock, self).__init__()

        self.planes = planes
        # Top-down layers, use nn.ConvTranspose2d to replace
        # nn.Conv2d+F.upsample?
        self.toplayer1 = nn.Conv2d(
            2048,
            planes,
            kernel_size=1,
            stride=1,
            padding=0)  # Reduce channels
        self.toplayer2 = nn.Conv2d(
            512, planes, kernel_size=3, stride=1, padding=1)
        self.toplayer3 = nn.Conv2d(
            512, planes, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(
            1024, planes, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(
            512, planes, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(
            x,
            size=(
                H,
                W),
            mode='bilinear',
            align_corners=True) + y

    def forward(self, c3, c4, c5):
        # Top-down
        p5 = self.toplayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.toplayer2(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.toplayer3(p3)

        return p3, p4, p5


if __name__ == '__main__':
    model = ResNetFPN()

    x = torch.randn((2, 1, 64, 256))
    y = model(x)
    print(y.shape)
