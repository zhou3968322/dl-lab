# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/14


def get_model_name(name):
    name_list = name.split('_')
    class_name = ""
    for name in name_list:
        class_name += "{}{}".format(name[0].upper(), name[1:])
    return class_name


def get_trainer_name(name):
    return "{}{}Trainer".format(name[0].upper(), name[1:])



