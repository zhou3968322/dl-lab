# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/2/1
from torch import nn
from networks.components.lstm import BidirectionalLSTM


class PaperRnn(nn.Module):

    def __init__(self, feature_extractor_output, hidden_size):
        super(PaperRnn, self).__init__()
        self.rnn = nn.Sequential(
            BidirectionalLSTM(feature_extractor_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
        )

    def forward(self, x):
        output = self.rnn(x)
        return output


