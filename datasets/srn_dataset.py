# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/30
import os, json
from datasets.transforms import default_transform
from torch.utils.data.dataset import Dataset
from utils.dataset_util import make_dataset
from PIL import Image, ImageOps
import numpy as np


class SrnDataset(Dataset):
    """A dataset class for noise, gt, mask image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B, C}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, data_root, n_levels=3, scale=0.5, resample=None, phase="train", max_dataset_size=np.inf):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(SrnDataset, self).__init__()
        self.dir_AB = os.path.join(data_root, phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, max_dataset_size))  # get image paths
        # random.shuffle(self.AB_paths)
        self.dataset_len = len(self.AB_paths)
        self.n_levels = n_levels
        self.scale = scale
        if resample is None:
            self.resample = Image.BILINEAR
        else:
            self.resample

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

        # split AB image into A and B
        w, h = AB.size
        w3 = int(w / 3)
        blurry_img = AB.crop((0, 0, w3, h))
        # B = AB.crop((w3, 0, 2 * w3, h)).convert("L")
        clear_img = AB.crop((w3, 0, 2 * w3, h))

        current_transform = default_transform()

        data = {}
        blur_key = "blur"
        clear_key = "clear"

        for i in range(self.n_levels):
            if i > 0:
                blur = blurry_img
                clear = clear_img
            else:
                scale = self.scale ** (i + 1)
                nw = int(w3 * scale)
                nh = int(h * scale)
                blur = blurry_img.resize((nw, nh), resample=self.resample)
                clear = clear_img.resize((nw, nh), resample=self.resample)
            data.update({
                "{}{}".format(blur_key, i): blur,
                "{}{}".format(clear_key, i): clear,
            })

        return data

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.dataset_len
