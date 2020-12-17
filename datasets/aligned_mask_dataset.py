# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/14
import os, json
from datasets.transforms import default_transform
from torch.utils.data.dataset import Dataset
from utils.dataset_util import make_dataset
from PIL import Image, ImageOps
import numpy as np
import random
from utils.img_util import convert_poly_to_rect


def _handler_mask_data(mask_data, crop_size=65):
    # 从mask中随机抠出一个64x64的框
    mask_data = np.array(mask_data)
    crop_box = []
    for mask_polys in mask_data:
        mask_rect = convert_poly_to_rect(mask_polys)
        mask_height = mask_rect[3] - mask_rect[1]
        mask_width = mask_rect[2] - mask_rect[0]
        if mask_width >= crop_size:
            xs = random.randint(0,mask_width - 64)
            xe = xs + crop_size
        elif mask_rect[2] >= crop_size:
            xs = mask_rect[2] - crop_size
            xe = mask_rect[2]
        else:
            xs = mask_rect[0]
            xe = mask_rect[0] + crop_size
        if mask_height >= crop_size:
            ys = random.randint(0,mask_height - crop_size)
            ye = ys + crop_size
        elif mask_rect[3] >= crop_size:
            ys = mask_rect[3] - crop_size
            ye = mask_rect[3]
        else:
            ys = mask_rect[1]
            ye = mask_rect[1] + crop_size
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
        self.dataset_len = len(self.AB_paths)
        mask_paths = []
        for ab_path in self.AB_paths:
            ab_name = os.path.basename(ab_path)
            mask_name = "{}_{}.json".format(phase, ab_name.rsplit('.', 1)[0])
            mask_path = os.path.join(mask_dir, mask_name)
            mask_paths.append(mask_path)
            assert os.path.isfile(mask_path)
        self.mask_paths = mask_paths
        random.seed(30)

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
        crop_box = random.choice(_handler_mask_data(mask_data))
        x0, y0, x1, y1 = crop_box
        # split AB image into A and B
        w, h = AB.size
        w3 = int(w / 3)
        A = AB.crop((0, 0, w3, h))
        B = AB.crop((w3, 0, 2 * w3, h))
        mask = AB.crop((2 * w3, 0, w, h)).convert("L")

        current_transform = default_transform()
        gray_transform = default_transform(grayscale=True)

        A = current_transform(A)
        B = current_transform(B)
        noise_mask = gray_transform(mask)
        # 初期训练的时候 mask_img > 0 ，并且不将mask填充. （这里对mask的处理0.2其实可以作为一个随机数）
        # 到后期训练的时候需要将mask在input中填充（这是为了捕捉到更多结构信息，而非颜色信息）
        # 网路需要学到颜色信息。
        # normalize 防止如果包含resize的情况, 插值的问题
        noise_mask[noise_mask >= -0.2] = 1.0
        noise_mask[noise_mask < 0.2] = -1.0

        A_local = A[:, x0: x1, y0:y1]
        B_local = B[:, x0: x1, y0:y1]
        noise_mask_local = noise_mask[:, x0: x1, y0:y1]
        return {'A': A, 'B': B, "noise_mask": noise_mask, "A_local": A_local,
                "B_local": B_local, "noise_mask_local": noise_mask_local}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
