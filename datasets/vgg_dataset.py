# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/1/20

from torch.utils.data.dataset import Dataset
from utils.dataset_util import make_dataset
from PIL import Image, ImageOps
import numpy as np
import random
from torchvision import transforms


class VggDataset(Dataset):
    """A dataset class for noise, gt, mask image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B, C}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, data_root, max_dataset_size=np.inf):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.AB_paths = sorted(make_dataset(data_root, max_dataset_size))  # get image paths
        self.transform = transforms.Compose([transforms.ToTensor()])
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
        w2 = int(w / 2)
        if w2 < 1024 or h < 1024:
            return self.__getitem__(index + 1)
        crop_x = random.randint(0, w2 - 1024)
        crop_y = random.randint(0, h - 1024)
        img_source = self.transform(AB.crop((crop_x, crop_y, crop_x + 1024, crop_y + 1024)))
        img_target = self.transform(AB.crop((w2 + crop_x, crop_y, w2 + crop_x + 1024, crop_y + 1024)))
        return {'img_source': img_source, 'img_target': img_target,
                "AB_path": AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

