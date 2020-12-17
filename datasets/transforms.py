# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/14
from PIL import Image
import torchvision.transforms as transforms


def default_transform(grayscale=False, convert=True):
    transform_list = []
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def default_resize_transform(load_size=(512, 512), grayscale=False, convert=True):
    transform_list = []
    osize = [load_size[0], load_size[1]]
    transform_list.append(transforms.Resize(osize, Image.ANTIALIAS))
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)