# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/14
from utils.common_util import get_model_name
import models, datasets
from utils.log import logger
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class MedfeTrainer(object):

    def __init__(self, config):
        self.experiment_name = config.pop('name')
        self.start_epoch = config["trainer"]["scheduler"]["start_epoch"]
        self.niter = config["trainer"]["scheduler"]["niter"]
        self.niter_decay = config["trainer"]["scheduler"]["niter_decay"]
        model_name = get_model_name(config["arch"].pop("type"))
        self.model = getattr(models, model_name)(config)
        logger.info("model init success")
        tensorboard_log_dir = config["tensorboard"]["log_dir"]
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir, comment=self.experiment_name)
        dataset_args = config["datasets"]["train"]["dataset"]
        dataset_type = dataset_args["type"]
        dataset_init_args = dataset_args["args"]
        dataset = getattr(datasets, dataset_type)(**dataset_init_args)
        data_loader_args = config["datasets"]["train"]["loader"]
        train_loader = DataLoader(dataset=dataset, **data_loader_args)
        self.train_loader = train_loader

    def train(self):
        for epoch in range(self.start_epoch, self.niter + self.niter_decay + 1):
            self.train_one_epoch(epoch)

    def train_one_epoch(self, epoch):
        epoch_stat_time = time.time()
        for train_data in self.train_loader:
            self.model.train(train_data)

