# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/14
import os, json
from datasets.transforms import default_transform
from torch.utils.data.dataset import Dataset
from utils.dataset_util import make_dataset
from PIL import Image, ImageOps
import numpy as np


class AlignedDataset(Dataset):
    """A dataset class for noise, gt, mask image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B, C}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, data_root, phase="train", max_dataset_size=np.inf):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(AlignedDataset, self).__init__()
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

        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w3, 0, 2 * w3, h)).convert("L")
        B = AB.crop((w2, 0, w, h))

        current_transform = default_transform()

        A = current_transform(A)
        B = current_transform(B)
        return {'A': A, 'B': B,
                "AB_path": AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
