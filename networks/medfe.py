# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/14
import torch
import torch.nn as nn
import functools


# Define the resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=False),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=False),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# define the Encoder unit
class UnetSkipConnectionEBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, kernel_size=4, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionEBlock, self).__init__()
        if kernel_size == 4:
            downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                                 stride=2, padding=1)
        else:
            downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=8,
                                 stride=4, padding=2)

        downrelu = nn.LeakyReLU(0.2, True)

        downnorm = norm_layer(inner_nc, affine=True)
        if outermost:
            down = [downconv]
            model = down
        elif innermost:
            down = [downrelu, downconv]
            model = down
        else:
            down = [downrelu, downconv, downnorm]
            if use_dropout:
                model = down + [nn.Dropout(0.5)]
            else:
                model = down
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UnetSkipConnectionDBlock(nn.Module):
    def __init__(self, inner_nc, outer_nc, kernel_size=4, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionDBlock, self).__init__()
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)
        if kernel_size == 8:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=8, stride=4,
                                        padding=2)
        else:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)

        if outermost:
            up = [uprelu, upconv, nn.Tanh()]
            model = up
        elif innermost:
            up = [uprelu, upconv, upnorm]
            model = up
        else:
            up = [uprelu, upconv, upnorm]
            model = up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, res_num=4, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Encoder, self).__init__()

        # construct unet structure
        Encoder_1 = UnetSkipConnectionEBlock(input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                             outermost=True)
        Encoder_2 = UnetSkipConnectionEBlock(ngf, ngf * 2, kernel_size=8, norm_layer=norm_layer,
                                             use_dropout=use_dropout)
        Encoder_3 = UnetSkipConnectionEBlock(ngf * 2, ngf * 4, kernel_size=8, norm_layer=norm_layer,
                                             use_dropout=use_dropout)
        Encoder_4 = UnetSkipConnectionEBlock(ngf * 4, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout)
        Encoder_5 = UnetSkipConnectionEBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout)
        Encoder_6 = UnetSkipConnectionEBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout,
                                             innermost=True)

        blocks = []
        for _ in range(res_num):
            block = ResnetBlock(ngf * 8, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.Encoder_1 = Encoder_1
        self.Encoder_2 = Encoder_2
        self.Encoder_3 = Encoder_3
        self.Encoder_4 = Encoder_4
        self.Encoder_5 = Encoder_5
        self.Encoder_6 = Encoder_6
        self.device = torch.device("cuda:0")

    def reset_device(self, device):
        self.device = device

    def forward(self, input):
        if self.device != input.device:
            input = input.to(self.device)
        y_1 = self.Encoder_1(input)
        y_2 = self.Encoder_2(y_1)
        y_3 = self.Encoder_3(y_2)
        y_4 = self.Encoder_4(y_3)
        y_5 = self.Encoder_5(y_4)
        y_6 = self.Encoder_6(y_5)
        y_7 = self.middle(y_6)

        return y_1, y_2, y_3, y_4, y_5, y_7


class Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Decoder, self).__init__()

        # construct unet structure
        Decoder_1 = UnetSkipConnectionDBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout,
                                             innermost=True)
        Decoder_2 = UnetSkipConnectionDBlock(ngf * 16, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout)
        Decoder_3 = UnetSkipConnectionDBlock(ngf * 16, ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout)
        Decoder_4 = UnetSkipConnectionDBlock(ngf * 8, ngf * 2, kernel_size=8, norm_layer=norm_layer, use_dropout=use_dropout)
        Decoder_5 = UnetSkipConnectionDBlock(ngf * 4, ngf, kernel_size=8, norm_layer=norm_layer, use_dropout=use_dropout)
        Decoder_6 = UnetSkipConnectionDBlock(ngf * 2, output_nc, norm_layer=norm_layer, use_dropout=use_dropout,
                                             outermost=True)

        self.Decoder_1 = Decoder_1
        self.Decoder_2 = Decoder_2
        self.Decoder_3 = Decoder_3
        self.Decoder_4 = Decoder_4
        self.Decoder_5 = Decoder_5
        self.Decoder_6 = Decoder_6
        self.device = torch.device("cuda:0")

    def reset_device(self, device):
        self.device = device

    def forward(self, input_1, input_2, input_3, input_4, input_5, input_6):
        if self.device != input_1.device:
            input_1 = input_1.to(self.device)
        if self.device != input_2.device:
            input_2 = input_2.to(self.device)
        if self.device != input_3.device:
            input_3 = input_3.to(self.device)
        if self.device != input_4.device:
            input_4 = input_4.to(self.device)
        if self.device != input_5.device:
            input_5 = input_5.to(self.device)
        if self.device != input_6.device:
            input_6 = input_6.to(self.device)
        y_1 = self.Decoder_1(input_6)
        y_2 = self.Decoder_2(torch.cat([y_1, input_5], 1))
        y_3 = self.Decoder_3(torch.cat([y_2, input_4], 1))
        y_4 = self.Decoder_4(torch.cat([y_3, input_3], 1))
        y_5 = self.Decoder_5(torch.cat([y_4, input_2], 1))
        y_6 = self.Decoder_6(torch.cat([y_5, input_1], 1))
        out = y_6

        return out
