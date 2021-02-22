# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/2/1
import networks
import os
import torch
from torch.functional import F
from utils.dl_util import init_net
from utils.dl_util import get_scheduler, get_optimizer, set_requires_grad, tensor2im
from models.base_model import BaseModel
from torch.nn import init
from utils.converter import SRNConverter
from utils.averager import Averager
import torchvision
from collections import OrderedDict


def srn_init_func(m):
    classname = m.__class__.__name__
    if 'localization_fc2' in classname:
        print(f'Skip {classname} as it is already initialized')
    elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.2)
        init.constant(m.bias.data, 0.0)


def cal_loss(pred, gold, PAD, smoothing='1'):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing == '0':
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(0)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    elif smoothing == '1':
        loss = F.cross_entropy(pred, gold, ignore_index=PAD)
    else:
        # loss = F.cross_entropy(pred, gold, ignore_index=PAD)
        loss = F.cross_entropy(pred, gold)

    return loss


def cal_performance(preds, gold, PAD, smoothing='1'):
    ''' Apply label smoothing if needed '''

    loss = 0.
    n_correct = 0
    weights = [1.0, 0.15, 2.0]
    for ori_pred, weight in zip(preds, weights):
        pred = ori_pred.view(-1, ori_pred.shape[-1])
        # debug show
        # t_gold = gold.view(ori_pred.shape[0], -1)
        # t_pred_index = ori_pred.max(2)[1]

        tloss = cal_loss(pred, gold, PAD, smoothing=smoothing)
        if torch.isnan(tloss):
            print('have nan loss')
            continue
        else:
            loss += tloss * weight

        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)
        n_correct = pred.eq(gold)
        non_pad_mask = gold.ne(PAD)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


class Srn(BaseModel):

    def __init__(self, config):
        super(Srn, self).__init__(config)
        arch_config = config.pop('arch')
        self.batch_max_length = arch_config.get("batch_max_length", 40)
        self.converter = SRNConverter(charset_path=arch_config["charset_path"], batch_max_length=self.batch_max_length)
        self.alphabet_size = len(self.converter.character)
        model_names = []
        self.srn = getattr(getattr(networks, arch_config["srn"].pop("type")),
                           arch_config["srn"].pop("name"))(batch_max_length=self.batch_max_length,
                                                           alphabet_size=self.alphabet_size,
                                                           **arch_config["srn"])
        model_names.append("srn")

        if self.mode == "train":
            init_args = arch_config.pop("init_args")
            self.srn = init_net(self.srn, i_func=srn_init_func, **init_args)
            self.criterion = cal_performance
            optimizer_args = config["trainer"].pop("optimizer")
            scheduler_args = config["trainer"].pop("scheduler")
            self.optimizer_srn = get_optimizer(self.srn, **optimizer_args)
            self.optimizers = [self.optimizer_srn]
            schedulers = []
            for optimizer in self.optimizers:
                schedulers.append(get_scheduler(optimizer, **scheduler_args))
            self.schedulers = schedulers
            self.grad_clip = config["trainer"].get("grad_clip", 5)
            self.loss_avg = Averager()

        if self.mode == "train" and self.continue_train:
            load_epoch = config["trainer"]["load_epoch"]
            self.load_networks(load_epoch)
        elif self.mode in ["predict", "eval"]:
            load_epoch = config["predictor"]["load_epoch"]
            self.load_networks(load_epoch)
        self.print_networks()

    def train(self, data):
        images = data["images"].to(self.device)
        text, length = self.converter.encode(data["labels"])
        text = text.to(self.device)
        length = length.to(self.device)
        preds = self.srn(images, text=None)
        cost, train_correct = self.criterion(preds, text, self.converter.PAD)
        self.optimizer_srn.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(self.srn.parameters(), self.grad_clip)  # gradient clipping with 5 (Default)
        self.optimizer_srn.step()
        self.loss_avg.add(cost)

    def check_data(self, data):
        for label in data["labels"]:
            if len(label) >= self.batch_max_length - 1:
                return False
        return True

    @torch.no_grad()
    def evaluate(self, data):
        images = data["images"].to(self.device)
        batch_size = len(data["labels"])
        length_for_pred = torch.cuda.IntTensor([self.batch_max_length] * batch_size)
        text, length = self.converter.encode(data["labels"])
        text = text.to(self.device)
        length = length.to(self.device)
        preds = self.srn(images, text=None)
        cost, train_correct = self.criterion(preds, text, self.converter.PAD)
        _, preds_index = preds[2].max(2)
        preds_str = self.converter.decode(preds_index, length_for_pred)
        labels = self.converter.decode(text, length)
        return cost, preds_str, labels

    @torch.no_grad()
    def inference(self, image):
        pass

    def set_mode(self, mode=None):
        if mode is None:
            mode = self.mode
        if mode == "eval":
            self.srn.eval()
            if self.mode == "train":
                self.srn.eval()
        else:
            self.srn.train()
            if self.mode == "train":
                self.srn.train()

    def get_current_visuals(self):
        pass

    def get_current_errors(self):
        # show the current loss
        return {
            "loss_avg": self.loss_avg.val()
        }
