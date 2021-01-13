# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/1/7

import os, json
from datasets.transforms import default_transform
from torch.utils.data.dataset import Dataset
from utils.dataset_util import make_dataset
from PIL import Image, ImageOps
import numpy as np
import random
from utils.img_util import convert_poly_to_rect


class Pix2PixDataset(Dataset):
    """A dataset class for noise, gt, mask image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B, C}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, data_root, phase="train", max_dataset_size=np.inf):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(Pix2PixDataset, self).__init__()
        self.dir_AB = os.path.join(data_root, phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, max_dataset_size))  # get image paths
        # random.shuffle(self.AB_paths)
        self.dataset_len = len(self.AB_paths)

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
        w, h = AB.size
        w3 = int(w / 3)
        A = AB.crop((0, 0, w3, h))
        # B = AB.crop((w3, 0, 2 * w3, h)).convert("L")
        mask = AB.crop((2 * w3, 0, w, h))
        mask, g, b = mask.split()
        current_transform = default_transform()
        gray_transform = default_transform(grayscale=True)

        A = current_transform(A)
        noise_mask = gray_transform(mask)
        return {'A': A, 'B': noise_mask,
                "AB_path": AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
