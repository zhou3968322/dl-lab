# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/2/1
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            # print(mask.shape, attn.shape, v.shape)
            attn = attn.masked_fill(mask, -1e9)

        attn = self.softmax(attn)       # 第3个维度为权重
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn