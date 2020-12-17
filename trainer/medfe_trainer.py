# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/14
from utils.common_util import get_model_name
import models, datasets
from utils.log import logger
import time, torch, torchvision
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
        self.display_freq = config["tensorboard"]["display_freq"]
        self.print_freq = config["tensorboard"]["print_freq"]
        self.save_epoch_freq = config["tensorboard"].get("save_epoch_freq", 0)
        self.save_step_freq = config["tensorboard"].get("save_step_freq", 0)

        dataset_args = config["datasets"]["train"]["dataset"]
        dataset_type = dataset_args["type"]
        dataset_init_args = dataset_args["args"]
        dataset = getattr(datasets, dataset_type)(**dataset_init_args)
        data_loader_args = config["datasets"]["train"]["loader"]
        train_loader = DataLoader(dataset=dataset, **data_loader_args)
        self.train_loader = train_loader

        self.global_step = 0

    def train(self):
        for epoch in range(self.start_epoch, self.niter + self.niter_decay + 1):
            self.train_one_epoch(epoch)
        self.writer.close()

    def train_one_epoch(self, epoch):
        epoch_start_time = time.time()
        for train_data in self.train_loader:
            iter_start_time = time.time()
            self.model.train(train_data)
            self.global_step += 1
            if self.global_step % self.print_freq == 0:
                errors = self.model.get_current_errors()
                t_comp = time.time() - iter_start_time
                message = '(epoch: %d, steps: %d, time: %.3f) ' % (epoch, self.global_step, t_comp)
                logger.info(message)
                for key, value in errors.items():
                    message += '%s: %.5f ' % (key, value)
                    self.writer.add_scalar(key, errors[key], self.global_step)
            if self.global_step & self.display_freq == 0:
                visual_input = self.model.get_current_visuals()
                image_out = torch.cat(list(visual_input), 0)
                grid = torchvision.utils.make_grid(image_out)
                self.writer.add_image('epoch_(%d)_(%d)' % (epoch, self.global_step), grid, self.global_step)
            if self.save_epoch_freq == 0 and self.save_step_freq > 0 and self.global_step % self.save_step_freq == 0:
                logger.info('saving the model epoch:{}, step:{}'.format(epoch, self.global_step))
                self.model.save_networks(epoch)
        if self.save_epoch_freq > 0 and epoch % self.save_epoch_freq == 0:
            logger.info('saving the model at the end of epoch:{}'.format(epoch))
            self.model.save_networks(epoch)
        logger.info('End of epoch {} / {} \t Time Taken: {} sec'.format(epoch, self.niter + self.niter_decay,
                                                                        time.time() - epoch_start_time))
        self.model.update_learning_rate()


