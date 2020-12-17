# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/
import torch
import torch.nn as nn
import numpy as np
from utils.dl_util import cal_feat_mask
import torch.nn.functional as F


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, inputt):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        input = inputt[0]
        mask = inputt[1].float().to(inputt[1].device)
        # output W^T *x
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)
        # output_bias: W^T * X + B

        # 在进行部分卷积之后，进行Mask的更新。更新规则为：
        # 如果卷积（滑动）窗口对应的Mask值至少有一个对应的1，那么就更新卷积后对应位置Mask为1。
        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes.bool(), 1.0)
        # output_pre 即为 W^T* (M .* X) / sum(M) + b
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes.bool(), 0.0)
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes.bool(), 0.0)
        out = []
        out.append(output)
        out.append(new_mask)
        return out


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='leaky',
                 conv_bias=False, innorm=False, inner=False, outer=False):
        super().__init__()
        if sample == 'same-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 1, 2, bias=conv_bias)
        elif sample == 'same-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 1, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.InstanceNorm2d(out_ch, affine=True)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.innorm = innorm
        self.inner = inner
        self.outer = outer

    def forward(self, input):
        out = input
        if self.inner:
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])

        elif self.innorm:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
        elif self.outer:
            out = self.conv(out)
            out[0] = self.bn(out[0])
        else:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            if hasattr(self, 'activation'):
                out[0] = self.activation(out[0])
        return out


class ConvDown(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding=0, dilation=1, groups=1, bias=False, layers=1, activ=True):
        super().__init__()
        nf_mult = 1
        nums = out_c / 64
        sequence = []

        for i in range(1, layers + 1):
            nf_mult_prev = nf_mult
            if nums == 8:
                if in_c == 512:

                    nfmult = 1
                else:
                    nf_mult = 2

            else:
                nf_mult = min(2 ** i, 8)
            if kernel != 1:

                if activ == False and layers == 1:
                    sequence += [
                        nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                  kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                        nn.InstanceNorm2d(nf_mult * in_c)
                    ]
                else:
                    sequence += [
                        nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                  kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                        nn.InstanceNorm2d(nf_mult * in_c),
                        nn.LeakyReLU(0.2, True)
                    ]

            else:

                sequence += [
                    nn.Conv2d(in_c, out_c,
                              kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                    nn.InstanceNorm2d(out_c),
                    nn.LeakyReLU(0.2, True)
                ]

            if activ == False:
                if i + 1 == layers:
                    if layers == 2:
                        sequence += [
                            nn.Conv2d(nf_mult * in_c, nf_mult * in_c,
                                      kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                            nn.InstanceNorm2d(nf_mult * in_c)
                        ]
                    else:
                        sequence += [
                            nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                      kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                            nn.InstanceNorm2d(nf_mult * in_c)
                        ]
                    break

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class ConvUp(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, kernel,
                              stride, padding, dilation, groups, bias)
        self.bn = nn.InstanceNorm2d(out_c)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, size):
        out = F.interpolate(input=input, size=size, mode='bilinear')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class PCconv(nn.Module):
    def __init__(self, input_w=1024, nc=None, nw=None):
        super(PCconv, self).__init__()
        # input_w mush be 2^k, 64设置为
        ni = 6
        if nc is None:
            nc = [64, 128, 256, 512, 512, 512]
        if nw is None:
            nw = [512, 256, 128, 64, 32, 16]
        self.nc = nc
        self.nw = nw
        # 使用第三层作为mask信息的特征feature
        mask_w = nw[2]
        mask_c = nc[2]
        self.mask_w = mask_w
        self.conv_feat_mask_layer_num = int(np.log2(input_w / mask_w))
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        # feature sample 层, 分为上采样和下采样
        self.down_1 = ConvDown(nc[0], 2 * nc[0], 4, 2, padding=1, layers=int(np.log2(nw[0] / mask_w)))
        self.down_2 = ConvDown(nc[1], 2 * nc[1], 4, 2, padding=1, layers=int(np.log2(nw[1] / mask_w)))
        self.down_3 = ConvDown(nc[2], nc[2], 1, 1)
        # 上采样层基本一致
        self.up = ConvUp(512, mask_c, 1, 1)

        # feature 层融合后的下采样层
        self.down = ConvDown(3 * mask_c, mask_c, 1, 1)

        # texture和structure融合后的下采样层
        self.fuse = ConvDown(2 * mask_c, 2 * mask_c, 1, 1)

        # texture和structure融合后上采样和下采样到input channel中
        self.up_1 = ConvUp(2 * mask_c, nc[0], 1, 1)
        self.up_2 = ConvUp(2 * mask_c, nc[1], 1, 1)
        self.up_3 = ConvUp(2 * mask_c, nc[2], 1, 1)

        self.down_4 = ConvDown(2 * mask_c, 2 * mask_c, 4, 2, padding=1, layers=int(np.log2(mask_w / nw[3])))
        self.down_5 = ConvDown(2 * mask_c, 2 * mask_c, 4, 2, padding=1, layers=int(np.log2(mask_w / nw[4])))
        self.down_6 = ConvDown(2 * mask_c, 2 * mask_c, 4, 2, padding=1, layers=int(np.log2(mask_w / nw[5])))

        # partial conv层
        seuqence_3 = []
        seuqence_5 = []
        seuqence_7 = []
        for i in range(5):
            seuqence_3 += [PCBActiv(256, 256, innorm=True)]
            seuqence_5 += [PCBActiv(256, 256, sample='same-5', innorm=True)]
            seuqence_7 += [PCBActiv(256, 256, sample='same-7', innorm=True)]
        self.cov_3 = nn.Sequential(*seuqence_3)
        self.cov_5 = nn.Sequential(*seuqence_5)
        self.cov_7 = nn.Sequential(*seuqence_7)
        self.device = torch.device("cuda:0")

    def reset_device(self, device):
        self.device = device

    def forward(self, input, mask):
        for i in range(len(input)):
            if self.device != input[i].device:
                input[i] = input[i].to(self.device)
        if self.device != mask.device:
            mask = mask.to(self.device)

        mask = cal_feat_mask(mask, self.conv_feat_mask_layer_num, 1)
        # input[2]:256 32 32
        b, c, h, w = input[2].size()
        mask_1 = torch.add(torch.neg(mask.float()), 1)
        mask_1 = mask_1.expand(b, c, h, w)

        x_1 = self.activation(input[0])
        x_2 = self.activation(input[1])
        x_3 = self.activation(input[2])
        x_4 = self.activation(input[3])
        x_5 = self.activation(input[4])
        x_6 = self.activation(input[5])
        # Change the shape of each layer and intergrate low-level/high-level features

        x_1 = self.down_1(x_1)
        x_2 = self.down_2(x_2)
        x_3 = self.down_3(x_3)
        x_4 = self.up(x_4, (self.mask_w, self.mask_w))
        x_5 = self.up(x_5, (self.mask_w, self.mask_w))
        x_6 = self.up(x_6, (self.mask_w, self.mask_w))

        # The first three layers are Texture/detail
        # The last three layers are Structure
        x_texture = torch.cat([x_1, x_2, x_3], 1)
        x_structure = torch.cat([x_4, x_5, x_6], 1)

        x_texture = self.down(x_texture)
        x_structure = self.down(x_structure)

        # Multi Scale PConv fill the Details
        x_texture_3 = self.cov_3([x_texture, mask_1])
        x_texture_5 = self.cov_5([x_texture, mask_1])
        x_texture_7 = self.cov_7([x_texture, mask_1])
        x_texture_fuse = torch.cat([x_texture_3[0], x_texture_5[0], x_texture_7[0]], 1)
        x_texture_fi = self.down(x_texture_fuse)

        # Multi Scale PConv fill the Structure
        x_structure_3 = self.cov_3([x_structure, mask_1])
        x_structure_5 = self.cov_5([x_structure, mask_1])
        x_structure_7 = self.cov_7([x_structure, mask_1])
        x_structure_fuse = torch.cat([x_structure_3[0], x_structure_5[0], x_structure_7[0]], 1)
        x_structure_fi = self.down(x_structure_fuse)

        x_cat_fuse = self.fuse(torch.cat([x_structure_fi, x_texture_fi], 1))

        x_1 = self.up_1(x_cat_fuse, (self.nw[0], self.nw[0])) + input[0]
        x_2 = self.up_2(x_cat_fuse, (self.nw[1], self.nw[1])) + input[1]
        x_3 = self.up_3(x_cat_fuse, (self.nw[2], self.nw[2])) + input[2]
        x_4 = self.down_4(x_cat_fuse) + input[3]
        x_5 = self.down_5(x_cat_fuse) + input[4]
        x_6 = self.down_6(x_cat_fuse) + input[5]

        out = [x_1, x_2, x_3, x_4, x_5, x_6]
        return out
