# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/2/18
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
LONG_NUMS = 0


# TODO
def padding_and_resize(img, target_w, target_h, interpolation, keep_ratio=True, nc=1):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if nc == 1:
        img = img.convert('L')
    w, h = img.size
    im_ratio = w / h
    target_ratio = target_w / target_h
    if im_ratio > target_ratio:
        # if w > target_w: # w > target_w的时候 按照宽来resize, paste, 否则是能直接paste的
        th = min(int(target_w / im_ratio), target_h)
        img = img.resize((target_w, th), interpolation)
    else:
        # if h > target_h: # h > target_h的时候，按照高度来resize, paste，否则使能直接paste的
        tw = min(int(target_h * im_ratio), target_w)
        img = img.resize((tw, target_h), interpolation)
    if nc == 1:
        new_img = Image.new('L', (target_w, target_h), 255)
    else:
        new_img = Image.new('RGB', (target_w, target_h), (255, 255, 255))
    new_img.paste(img, (0, 0))
    return new_img


class PaddingResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR, keep_ratio=True, nc=1):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.keep_ratio = keep_ratio
        self.nc = nc

    def __call__(self, img):
        # TODO
        img = padding_and_resize(img, self.size[0], self.size[1], self.interpolation, self.keep_ratio, self.nc)
        # img = padding_and_resize_cv2(img, self.size[0], self.size[1], interpolation=cv2.INTER_CUBIC,
        #                              keep_ratio=self.keep_ratio, nc=self.nc)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class PaddingCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=True, nc=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.nc = nc
        self.transform = PaddingResizeNormalize((imgW, imgH), keep_ratio=self.keep_ratio, nc=self.nc)

    def __call__(self, batch):
        images = [i['img'] for i in batch]
        labels = [i['label'] for i in batch]

        images = [self.transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        return {
            "images": images,
            "labels": labels,
        }

