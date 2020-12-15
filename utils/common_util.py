# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/14


def get_model_name(name):
    return "{}{}".format(name[0].upper(), name[1:])


def get_trainer_name(name):
    return "{}{}Trainer".format(name[0].upper(), name[1:])
