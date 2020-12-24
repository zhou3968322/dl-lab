# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/14


# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 22:00
# @Author  : zhoujun

import argparse
import anyconfig
import predictor
from utils.common_util import get_predictor_name


def init_args():
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--config_file', default='config/gan/medfe_predict.yaml', type=str)
    parser.add_argument('--predictor_name', default='medfe', type=str)
    args = parser.parse_args()
    return args


def main(args):
    config = anyconfig.load(open(args.config_file, 'rb'))
    experiment_predictor = getattr(predictor, get_predictor_name(args.predictor_name))(config)
    experiment_predictor.predict()


if __name__ == '__main__':
    config_args = init_args()
    main(config_args)

