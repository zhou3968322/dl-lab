# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/15
import abc, os
import torch
from utils.log import logger
from utils.dl_util import print_network


class BaseModel(object):

    def __init__(self, config):
        self.model_names = []
        self.mode = config.pop("mode")
        self.continue_train = False
        self.device = torch.device('cuda') if torch.cuda.device_count() >= 1 else torch.device('cpu')
        if config["mode"] == "train":
            self.save_dir = config["trainer"].pop("save_dir")
            self.continue_train = config["trainer"].pop("continue_train")


    @abc.abstractmethod
    def print_networks(self):
        pass

    @abc.abstractmethod
    def save_networks(self):
        pass

    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '{}_net_epoch_{}.pth' % (name, which_epoch)
                load_path = os.path.join(self.save_dir, load_filename)

                net = getattr(self, 'net' + name)
                optimize = getattr(self, 'optimizer_' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path.replace('\\', '/'), map_location=str(self.device))
                optimize.load_state_dict(state_dict['optimize'])
                net.load_state_dict(state_dict['net'])

    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '{}_net_epoch_{}.pth' % (name, which_epoch)
                save_path = os.path.join(self.save_dir, save_filename).replace('\\', '/')
                net = getattr(self, name)
                optimize = getattr(self, 'optimizer_' + name)

                if torch.cuda.is_available():
                    torch.save({'net': net.module.cpu().state_dict(), 'optimize': optimize.state_dict()}, save_path)
                    net.cuda()
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        logger.info('current learning rate = %.7f' % lr)

    def print_networks(self):
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            print_network(net)