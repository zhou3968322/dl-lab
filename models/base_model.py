# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/15
import abc, os
import torch
from utils.log import logger
from utils.dl_util import print_network, get_gpu_ids


class BaseModel(object):

    def __init__(self, config):
        self.model_names = []
        self.mode = config.pop("mode")
        self.continue_train = False
        self.gpu_ids = get_gpu_ids(config["arch"]["init_args"].get("gpu_ids", "0"))
        self.device = torch.device("cuda") if len(self.gpu_ids) > 0 else torch.device("cpu")
        self.random_seed = config.get("random_seed", 30)
        torch.manual_seed(self.random_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(self.random_seed)
        if self.mode == "train":
            self.save_dir = config["trainer"].pop("save_dir")
            os.makedirs(self.save_dir, exist_ok=True)
            self.continue_train = config["trainer"].pop("continue_train")
        elif self.mode in ["predict", "eval"]:
            self.save_dir = config["predictor"].pop("model_dir")

    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '{}_net_epoch{}.pth'.format(name, which_epoch)
                load_path = os.path.join(self.save_dir, load_filename)

                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path.replace('\\', '/'), map_location=str(self.device))
                if self.mode == "train":
                    optimize = getattr(self, 'optimizer_' + name)
                    optimize.load_state_dict(state_dict['optimize'])
                net.load_state_dict(state_dict['net'])
                logger.info("load model_name:{} success".format(name))

    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '{}_net_epoch{}.pth'.format(name, which_epoch)
                save_path = os.path.join(self.save_dir, save_filename).replace('\\', '/')
                net = getattr(self, name)
                optimize = getattr(self, 'optimizer_' + name)
                if torch.cuda.is_available():
                    if isinstance(net, torch.nn.DataParallel):

                        torch.save({'net': net.module.state_dict(), 'optimize': optimize.state_dict()}, save_path)
                    else:
                        torch.save({'net': net.state_dict(), 'optimize': optimize.state_dict()}, save_path)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        logger.info('current learning rate = %.7f' % lr)

    def print_networks(self):
        for name in self.model_names:
            net = getattr(self, name)
            print_network(net)

    def reset_device(self):
        for name in self.model_names:
            net = getattr(self, name)
            if len(self.gpu_ids) > 0:
                assert (torch.cuda.is_available())
                if len(self.gpu_ids) > 1:
                    net = torch.nn.DataParallel(net, self.gpu_ids)
                else:
                    net.to(torch.device("cuda:{}".format(self.gpu_ids[0])))
