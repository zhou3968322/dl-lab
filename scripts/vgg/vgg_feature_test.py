# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/1/20

import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(__file__, "../../../"))
sys.path.append(PROJECT_DIR)

import argparse
from torch import nn
import torch
from datasets.vgg_dataset import VggDataset
from networks.backbone.vgg import VGG16
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict


VGG_FEATURE_NAMES = ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1',
                     'relu3_2', 'relu3_3', 'relu4_1', 'relu4_2', 'relu4_3',
                     'relu5_1', 'relu5_2', 'relu5_3']


class VggCollector(object):

    def __init__(self, feature_names=None):
        """
        all feature names is:
            'relu1_1','relu1_2','relu2_1','relu2_2','relu3_1','relu3_2',
            'relu3_3','max_3','relu4_1','relu4_2','relu4_3','relu5_1',
            'relu5_2','relu5_3'
        """
        super(VggCollector, self).__init__()
        self.criterion = torch.nn.MSELoss()
        if feature_names is None:
            self.feature_names = ["relu1_1", "relu2_1", "relu3_1", "relu4_1"]
        else:
            self.feature_names = feature_names
        self.sum_count = 0
        feature_data_list = []
        for feature_name in self.feature_names:
            feature_data_list.append(
                {
                    "sum": 0,
                    "mean": 0.0,
                    "val": 0.0
                }
            )
        self.feature_data_list = feature_data_list

    def __call__(self, x_vgg, y_vgg):
        for i, feature_name in enumerate(self.feature_names):
            feature_loss = 1.0 * self.criterion(x_vgg[feature_name], y_vgg[feature_name])
            old_sum = self.feature_data_list[i]["sum"]
            old_mean = self.feature_data_list[i]["mean"]
            self.feature_data_list[i]["val"] = feature_loss.data
            mean_val = (old_mean * old_sum + feature_loss.data) / (old_sum + 1)
            self.feature_data_list[i]["mean"] = mean_val
            self.feature_data_list[i]["sum"] += old_sum + 1
        return self.feature_data_list


def run(args):
    tensorboard_log_dir = args.tensorboard_log_dir
    if not os.path.isdir(tensorboard_log_dir):
        tensorboard_log_dir = os.path.join(PROJECT_DIR, tensorboard_log_dir)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_dir, comment="test_vgg")
    dataset = VggDataset(data_root=args.data_root)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                              shuffle=args.shuffle, num_workers=args.num_workers)
    vgg = VGG16().cuda()
    vgg_collector = VggCollector()
    device = torch.device("cuda:0")
    global_step = 1
    for data in train_loader:
        x = data["img_source"].to(device)
        y = data["img_target"].to(device)
        x_vgg = vgg(x)
        y_vgg = vgg(y)
        vgg_collector(x_vgg, y_vgg)
        if global_step % args.display_freq == 0:
            for i, feature_name in enumerate(vgg_collector.feature_names):
                feature_data = vgg_collector.feature_data_list[i]
                for key, value in feature_data.items():
                    writer.add_scalar("{}_{}".format(feature_name, key), value, global_step)
        global_step += 1


def main():
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--data_root', default="/data_ssd/ocr/zhoubingcheng/gan_datasets/"
                                               "gan_document_aligned_data/train/aligned_mask_img", type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--display_freq', default=10, type=int)
    parser.add_argument('--tensorboard_log_dir', default="./tensorboard/vgg")
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
