# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/2
import torch.nn as nn
from networks.attention import Attention
from networks.lstm import BidirectionalLSTM
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


def _init_paper_cnn(n_c=3, img_h=32, leakyRelu=False):
    ks = [3, 3, 3, 3, 3, 3, (img_h // 16, 2)]
    ps = [1, 1, 1, 1, 1, 1, 0]
    ss = [1, 1, 1, 1, 1, 1, 1]
    nm = [64, 128, 256, 256, 512, 512, 512]
    cnn = nn.Sequential()

    def conv_relu(i, batch_normalization=False):
        n_in = n_c if i == 0 else nm[i - 1]
        n_out = nm[i]
        cnn.add_module('conv{0}'.format(i), nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i]))
        if batch_normalization:
            cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(n_out))
        if leakyRelu:
            cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
        else:
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

    conv_relu(0)
    cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
    conv_relu(1)
    cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
    conv_relu(2, True)
    conv_relu(3)
    cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
    conv_relu(4, True)
    conv_relu(5)
    cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
    conv_relu(6, True)  # 512x1x16
    return cnn


class PaperCRNN(nn.Module):

    def __init__(self, img_h, nc, n_class, nh, leakyRelu=False):
        super(PaperCRNN, self).__init__()
        assert img_h % 16 == 0, 'imgH has to be a multiple of 16'
        self.cnn = _init_paper_cnn(n_c=nc, img_h=img_h, leakyRelu=leakyRelu)
        self.rnn = nn.Sequential(BidirectionalLSTM(512, nh, nh),
                                 BidirectionalLSTM(nh, nh, n_class))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        output = F.log_softmax(output, dim=2)
        return output


class AttentionCRNN(nn.Module):

    def __init__(self, img_h, nc, nclass, nh, leakyRelu=False):
        super(AttentionCRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH must be a multiple of 16'

        self.cnn = resnet18(pretrained=True)

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh),
        )

        self.attention = Attention(nh, nh, nclass)

    def forward(self, input, length):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        rnn = self.rnn(conv)
        output = self.attention(rnn, length)
        return output
