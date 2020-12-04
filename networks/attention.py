# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings=128):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size,bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size+num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_embeddings = num_embeddings
        self.processed_batches = 0

    def forward(self, prev_hidden, feats, cur_embeddings):
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size

        feats_proj = self.i2h(feats.view(-1,nC))
        prev_hidden_proj = self.h2h(prev_hidden).view(1,nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(-1, hidden_size)
        emition = self.score(torch.tanh(feats_proj + prev_hidden_proj).view(-1, hidden_size)).view(nT,nB).transpose(0,1)
        self.processed_batches = self.processed_batches + 1

        if self.processed_batches % 10000 == 0:
            print('processed_batches = %d' %(self.processed_batches))

        alpha = F.softmax(emition) # nB * nT
        if self.processed_batches % 10000 == 0:
            print('emition ', list(emition.data[0]))
            print('alpha ', list(alpha.data[0]))
        context = (feats * alpha.transpose(0,1).contiguous().view(nT,nB,1).expand(nT, nB, nC)).sum(0).squeeze(0) # nB * nC//感觉不应该sum，输出4×256
        context = torch.cat([context, cur_embeddings], 1)
        cur_hidden = self.rnn(context, prev_hidden)
        return cur_hidden, alpha


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.generator = nn.Linear(hidden_size, num_classes)
        self.processed_batches = 0

    def forward(self, feats, text_length):
        self.processed_batches = self.processed_batches + 1
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size
        assert(input_size == nC)
        assert(nB == text_length.numel())

        num_steps = text_length.data.max()
        num_labels = text_length.data.sum()

        output_hiddens = Variable(torch.zeros(num_steps, nB, hidden_size).type_as(feats.data))
        hidden = Variable(torch.zeros(nB,hidden_size).type_as(feats.data))
        max_locs = torch.zeros(num_steps, nB)
        max_vals = torch.zeros(num_steps, nB)
        for i in range(num_steps):
            hidden, alpha = self.attention_cell(hidden, feats)
            output_hiddens[i] = hidden
            if self.processed_batches % 500 == 0:
                max_val, max_loc = alpha.data.max(1)
                max_locs[i] = max_loc.cpu()
                max_vals[i] = max_val.cpu()
        if self.processed_batches % 500 == 0:
            print('max_locs', list(max_locs[0:text_length.data[0],0]))
            print('max_vals', list(max_vals[0:text_length.data[0],0]))
        new_hiddens = Variable(torch.zeros(num_labels, hidden_size).type_as(feats.data))
        b = 0
        start = 0
        for length in text_length.data:
            new_hiddens[start:start+length] = output_hiddens[0:length,b,:]
            start = start + length
            b = b + 1
        probs = self.generator(new_hiddens)
        return probs
