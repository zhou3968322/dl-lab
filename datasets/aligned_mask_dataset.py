# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/14
import os, json
from datasets.transforms import default_transform
from torch.utils.data.dataset import Dataset
from utils.dataset_util import make_dataset
from PIL import Image, ImageOps
import torch
import numpy as np
import random
from utils.img_util import convert_poly_to_rect


def _handler_mask_data(mask_data, crop_size=256, image_width=1024, image_height=1024):
    # 从mask中随机抠出一个64x64的框
    mask_data = np.array(mask_data)
    crop_box = []
    for mask_polys in mask_data:
        mask_rect = convert_poly_to_rect(mask_polys)
        mask_height = mask_rect[3] - mask_rect[1]
        mask_width = mask_rect[2] - mask_rect[0]
        if mask_rect[2] >= image_width:
            if image_width - mask_rect[0] > crop_size:
                xs = random.randint(mask_rect[0], image_width - crop_size)
                xe = xs + crop_size
            else:
                xs = image_width - crop_size
                xe = xs + crop_size
        elif mask_width >= crop_size:
            xs = random.randint(0,mask_width - crop_size)
            xe = xs + crop_size
        elif mask_rect[2] >= crop_size:
            xs = mask_rect[2] - crop_size
            xe = mask_rect[2]
        else:
            xs = mask_rect[0]
            xe = mask_rect[0] + crop_size
        if mask_rect[3] >= image_height:
            if image_height - mask_rect[1] > crop_size:
                ys = random.randint(mask_rect[1], image_height - crop_size)
                ye = ys + crop_size
            else:
                ys = image_height - crop_size
                ye = ys + crop_size
        elif mask_height >= crop_size:
            ys = random.randint(0,mask_height - crop_size)
            ye = ys + crop_size
        elif mask_rect[3] >= crop_size:
            ys = mask_rect[3] - crop_size
            ye = mask_rect[3]
        else:
            ys = mask_rect[1]
            ye = mask_rect[1] + crop_size
        if 0 < xs < xe < image_width and 0 < ys < ye < image_height:
            crop_box.append([xs, ys, xe, ye])
    return np.array(crop_box)


class AlignedMaskDataset(Dataset):
    """A dataset class for noise, gt, mask image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B, C}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, data_root, phase="train", max_dataset_size=np.inf):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(AlignedMaskDataset, self).__init__()
        self.dir_AB = os.path.join(data_root, phase)  # get the image directory
        mask_dir = os.path.join(data_root, "mask")
        self.AB_paths = sorted(make_dataset(self.dir_AB, max_dataset_size))  # get image paths
        # random.shuffle(self.AB_paths)
        self.dataset_len = len(self.AB_paths)
        mask_paths = []
        for ab_path in self.AB_paths:
            ab_name = os.path.basename(ab_path)
            mask_name = "{}_{}.json".format(phase, ab_name.rsplit('.', 1)[0])
            mask_path = os.path.join(mask_dir, mask_name)
            mask_paths.append(mask_path)
            assert os.path.isfile(mask_path)
        self.mask_paths = mask_paths

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index % self.dataset_len]
        try:
            AB = Image.open(AB_path).convert('RGB')
        except Exception as e:
            return self.__getitem__(index + 1)
        mask_path = self.mask_paths[index % self.dataset_len]
        with open(mask_path, "r") as fr:
            mask_data = json.loads(fr.read())

        # split AB image into A and B
        w, h = AB.size
        w3 = int(w / 3)
        A = AB.crop((0, 0, w3, h))
        # B = AB.crop((w3, 0, 2 * w3, h)).convert("L")
        B = AB.crop((w3, 0, 2 * w3, h))
        mask = AB.crop((2 * w3, 0, w, h))
        mask, g, b = mask.split()

        crop_boxes = _handler_mask_data(mask_data)
        if len(crop_boxes) > 0:
            crop_box = random.choice(crop_boxes)
        else:
            xs = random.randint(0, w3 - 256)
            ys = random.randint(0, h - 256)
            crop_box = np.array([xs, ys, xs + 256, ys + 256])
        current_transform = default_transform()
        gray_transform = default_transform(grayscale=True)

        A = current_transform(A)
        B = current_transform(B)
        noise_mask = gray_transform(mask)
        # 初期训练的时候 mask_img > 0 ，并且不将mask填充. （这里对mask的处理0.2其实可以作为一个随机数）
        # 到后期训练的时候需要将mask在input中填充（这是为了捕捉到更多结构信息，而非颜色信息）
        # 网路需要学到颜色信息。
        # normalize 防止如果包含resize的情况, 插值的问题
        # noise_mask[noise_mask >= -0.2] = 1.0
        # noise_mask[noise_mask < -0.2] = -1.0
        return {'A': A, 'B': B, "noise_mask": noise_mask, "crop_box": torch.tensor(crop_box),
                "AB_path": AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
