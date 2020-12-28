# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/14
import os
import glob
from natsort import natsorted


def get_model_name(name):
    name_list = name.split('_')
    class_name = ""
    for name in name_list:
        class_name += "{}{}".format(name[0].upper(), name[1:])
    return class_name


def get_trainer_name(name):
    return "{}{}Trainer".format(name[0].upper(), name[1:])


def get_predictor_name(name):
    return "{}{}Predictor".format(name[0].upper(), name[1:])


def get_file_list(folder_path: str, p_postfix: list = None, sub_dir: bool = True) -> list:
    """
    获取所给文件目录里的指定后缀的文件,读取文件列表目前使用的是 os.walk 和 os.listdir ，这两个目前比 pathlib 快很多
    :param filder_path: 文件夹名称
    :param p_postfix: 文件后缀,如果为 [.*]将返回全部文件
    :param sub_dir: 是否搜索子文件夹
    :return: 获取到的指定类型的文件列表
    """
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)
    if p_postfix is None:
        p_postfix = ['.jpg']
    if isinstance(p_postfix, str):
        p_postfix = [p_postfix]
    file_list = [x for x in glob.glob(folder_path + '/**/*.*', recursive=True) if
                 os.path.splitext(x)[-1] in p_postfix or '.*' in p_postfix]
    return natsorted(file_list)

