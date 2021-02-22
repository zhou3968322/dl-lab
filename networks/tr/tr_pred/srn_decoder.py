# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/2/1
import torch
from torch import nn
from networks.components.attention.base import PositionalEncoding
from networks.tr.tr_seq.transform_encoder import TransformEncoder


def get_pad_mask(seq, pad_idx):
    return (seq == pad_idx).unsqueeze(-2)


class Torch_transformer_encoder(nn.Module):
    '''
        use pytorch transformer for sequence learning

    '''

    def __init__(self, d_word_vec=512, n_layers=2, n_head=8, d_model=512, dim_feedforward=1024, n_position=256):
        super(Torch_transformer_encoder, self).__init__()

        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward)
        self.layer_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=self.layer_norm)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, cnn_feature, src_mask=None, return_attns=False):
        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.dropout(self.position_enc(cnn_feature))  # position embeding

        enc_output = self.encoder(enc_output)

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class PVAM(nn.Module):
    ''' Parallel Visual attention module 平行解码'''
    '''
    n_dim：512，阅读顺序序列编码的空间维度
    N_max_character: 25，单张图片最多有多少个字符
    n_position: cnn出来之后特征的序列长度
    '''

    def __init__(self, n_dim=512, N_max_character=25, n_position=256):
        super(PVAM, self).__init__()
        self.character_len = N_max_character

        self.f0_embedding = nn.Embedding(N_max_character, n_dim)

        self.w0 = nn.Linear(N_max_character, n_position)
        self.wv = nn.Linear(n_dim, n_dim)
        # first linear(512,25)
        self.we = nn.Linear(n_dim, N_max_character)

        self.active = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, enc_output):
        reading_order = torch.arange(self.character_len, dtype=torch.long, device=enc_output.device)
        reading_order = reading_order.unsqueeze(0).expand(enc_output.size(0), -1)  # (S,) -> (B, S)
        reading_order_embed = self.f0_embedding(reading_order)  # b,25,512

        t = self.w0(reading_order_embed.permute(0, 2, 1))  # b,512,256
        t = self.active(t.permute(0, 2, 1) + self.wv(enc_output))  # b,256,512
        # first linear(512,25)
        attn = self.we(t)  # b,256,25

        attn = self.softmax(attn.permute(0, 2, 1))  # b,25,256

        g_output = torch.bmm(attn, enc_output)  # b,25,512
        return g_output


class GSRM(nn.Module):
    # global semantic reasoning module
    '''
    n_dim：embed编码的特征空间维度
    n_class：embedding需要用到
    PAD：计算mask用到
    '''

    def __init__(self, n_dim=512, n_class=37, PAD=37 - 1, n_layers=4, n_position=25):
        super(GSRM, self).__init__()

        self.PAD = PAD
        self.argmax_embed = nn.Embedding(n_class, n_dim)

        self.transformer_units = TransformEncoder(n_layers=n_layers,
                                                  n_position=n_position)  # for global context information
        # self.transformer_units = Torch_transformer_encoder(n_layers=n_layers, n_position=n_position)

    def forward(self, e_out):
        '''
        e_out: b,25,37 | the output from PVAM3
        '''
        e_argmax = e_out.argmax(dim=-1)  # b, 25
        e = self.argmax_embed(e_argmax)  # b,25,512

        e_mask = get_pad_mask(e_argmax, self.PAD)  # b,25,1
        s = self.transformer_units(e, None)  # b,25,512

        return s


class SrnDecoder(nn.Module):
    # the wrapper of decoder layers
    '''
    n_dim: 特征空间维度
    n_class：字符种类
    N_max_character: 单张图最多只25个字符
    n_position：cnn输出的特征序列长度
    整个有三个部分的输出
    '''

    def __init__(self, n_dim=512, n_class=37, N_max_character=25, n_position=256, GSRM_layer=4):
        super(SrnDecoder, self).__init__()

        self.pvam = PVAM(N_max_character=N_max_character, n_position=n_position)
        self.w_e = nn.Linear(n_dim, n_class)  # output layer

        self.GSRM = GSRM(n_class=n_class, PAD=n_class - 1, n_dim=n_dim, n_position=N_max_character, n_layers=GSRM_layer)
        self.w_s = nn.Linear(n_dim, n_class)  # output layer

        self.w_f = nn.Linear(n_dim, n_class)  # output layer

    def forward(self, cnn_feature):
        '''cnn_feature: b,256,512 | the output from cnn'''

        g_output = self.pvam(cnn_feature)  # b,25,512
        e_out = self.w_e(g_output)  # b,25,37 ----> cross entropy loss  |  第一个输出

        s = self.GSRM(e_out)[0]  # b,25,512
        s_out = self.w_s(s)  # b,25,37f

        # TODO:change the add to gated unit
        f = g_output + s  # b,25,512
        f_out = self.w_f(f)

        return e_out, s_out, f_out
