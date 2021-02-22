# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/1/15

import os, json
from datasets.transforms import default_transform
from torch.utils.data.dataset import Dataset
from utils.dataset_util import make_dataset
from PIL import Image, ImageOps
import numpy as np
import random
from torchvision import transforms


class WdnetDataset(Dataset):
    """A dataset class for noise, gt, mask image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B, C}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, data_root, phase="train", max_dataset_size=np.inf):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(WdnetDataset, self).__init__()
        train_dir = os.path.join(data_root, phase)
        self.dir_AB = os.path.join(train_dir, "aligned_img")  # get the image directory
        self.dir_AB_mask = os.path.join(train_dir, "aligned_mask_img")
        self.AB_paths = sorted(make_dataset(self.dir_AB, max_dataset_size))  # get image paths
        random.shuffle(self.AB_paths)
        mask_paths = []
        for ab_path in self.AB_paths:
            ab_name = os.path.basename(ab_path)
            mask_path = os.path.join(self.dir_AB_mask, ab_name)
            mask_paths.append(mask_path)
            assert os.path.isfile(mask_path)
        self.mask_paths = mask_paths
        self.dataset_len = len(self.AB_paths)
        self.transform = transforms.Compose([transforms.ToTensor()])

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
        AB_mask_path = self.mask_paths[index % self.dataset_len]
        try:
            AB = Image.open(AB_path).convert('RGB')
            AB_mask = Image.open(AB_mask_path).convert('L')
        except Exception as e:
            return self.__getitem__(index + 1)
        w, h = AB.size
        w2 = int(w / 2)
        if w2 < 1024 or h < 1024:
            return self.__getitem__(index + 1)
        crop_x = random.randint(0, w2 - 1024)
        crop_y = random.randint(0, h - 1024)
        img_source = self.transform(AB.crop((crop_x, crop_y, crop_x + 1024, crop_y + 1024)))
        img_target = self.transform(AB.crop((w2 + crop_x, crop_y, w2 + crop_x + 1024, crop_y + 1024)))
        w = self.transform(AB_mask.crop((crop_x, crop_y, crop_x + 1024, crop_y + 1024)))
        mask = self.transform(AB_mask.crop((w2 + crop_x, crop_y, w2 + crop_x + 1024, crop_y + 1024)))
        alpha = self.transform(AB_mask.crop((2 * w2 + crop_x, crop_y, 2 * w2 + crop_x + 1024, crop_y + 1024)))
        return {'img_source': img_source, 'img_target': img_target,
                'w': w, "mask": mask, 'alpha': alpha,
                "AB_path": AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
