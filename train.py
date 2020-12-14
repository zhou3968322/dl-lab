# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/14


# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 22:00
# @Author  : zhoujun

import argparse
import os
import anyconfig


def init_args():
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--config_file', default='config/gan/structure_gan_train.yaml', type=str)
    parser.add_argument('--trainer_name', default='medfe', type=str)
    args = parser.parse_args()
    return args


def main(args):
    import torch
    config = anyconfig.load(open(args.config_file, 'rb'))


if __name__ == '__main__':
    config_args = init_args()
    main(config_args)

