# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/1/13
import networks
import os
import torch
from utils.dl_util import init_net
from networks.backbone.vgg import VGG16
from utils.dl_util import get_scheduler, get_optimizer, set_requires_grad, tensor2im
from models.base_model import BaseModel
from collections import OrderedDict
from metrics import metrics
from utils.log import logger


class VggLoss(torch.nn.Module):

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0], vgg_module=None):
        super(VggLoss, self).__init__()
        if vgg_module is None:
            self.vgg = VGG16().cuda()
        else:
            self.vgg = vgg_module
        self.criterion = torch.nn.MSELoss()
        self.weights = weights
        self.device = torch.device("cuda:0")

    def reset_device(self, device):
        self.device = device
        self.vgg = self.vgg.to(self.device)

    def __call__(self, x, y):
        # Compute features
        if x.device != self.device:
            x = x.to(self.device)
        if y.device != self.device:
            y = y.to(self.device)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        return content_loss


class Wdnet(BaseModel):

    def __init__(self, config):
        super(Wdnet, self).__init__(config)
        arch_config = config.pop('arch')
        init_args = arch_config.pop("init_args")
        model_names = []
        self.generator = getattr(getattr(networks, arch_config["generator"].pop("type")),
                                 arch_config["generator"].pop("name"))(**arch_config["generator"])
        self.generator = init_net(self.generator, **init_args)
        model_names.append("generator")
        logger.info("generator init success")
        self.device1 = torch.device("cuda:1")
        self.device2 = torch.device("cuda:2")
        self.generator.reset_post_process_device(self.device1)

        if self.mode == "train":
            self.discriminator = getattr(getattr(networks, arch_config["discriminator"].pop("type")),
                                         arch_config["discriminator"].pop("name"))(
                **arch_config["discriminator"])
            self.discriminator = init_net(self.discriminator, **init_args)
            model_names.append("discriminator")
            logger.info("discriminator init success")
        self.model_names = model_names

        if self.mode == "train":
            loss_args = config.get("loss", {})
            self.criterionVGG = VggLoss()
            self.criterionVGG.reset_device(self.device2)
            self.lambdaVgg = loss_args.get("lambdaVgg", 0.01)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionBCE = torch.nn.BCELoss()
            self.criterionMSE = torch.nn.MSELoss()
            self.lambdaMask = loss_args.get("lambdaMask", 50.0)
            self.lambdaW = loss_args.get("lambdaW", 10.0)
            self.lambdaAlpha = loss_args.get("lambdaAlpha", 10.0)
            self.lambdaI1 = loss_args.get("lambdaI1", 15.0)
            self.lambdaI2 = loss_args.get("lambdaI2", 35.0)

        # self.reset_device()

        if self.mode == "train":
            optimizer_args = config["trainer"].pop("optimizer")
            scheduler_args = config["trainer"].pop("scheduler")
            schedulers = []
            self.optimizer_generator = get_optimizer(self.generator, **optimizer_args)
            self.optimizer_discriminator = get_optimizer(self.discriminator, **optimizer_args)
            self.optimizers = [self.optimizer_generator, self.optimizer_discriminator]
            for optimizer in self.optimizers:
                schedulers.append(get_scheduler(optimizer, **scheduler_args))
            self.schedulers = schedulers

        if self.mode == "train" and self.continue_train:
            load_epoch = config["trainer"]["load_epoch"]
            self.load_networks(load_epoch)
        elif self.mode in ["predict", "eval"]:
            load_epoch = config["predictor"]["load_epoch"]
            self.load_networks(load_epoch)
        self.print_networks()
        self.iter = 0

    def train(self, data):
        self.input_a = data["img_source"].to(self.device)
        self.img_name = os.path.basename(data["AB_path"][0]).rsplit('.', 1)[0]
        self.input_b = data["img_target"].to(self.device)
        self.real_alpha = data["alpha"].to(self.device)
        self.real_w = data["w"].to(self.device)
        self.real_mask = data["mask"].to(self.device)
        self.fake_b, self.fake_mask, self.fake_alpha, self.fake_w, self.fake_i_w = self.generator(self.input_a)
        self.backward()
        self.iter += 1

    def backward(self):
        # update D
        if (self.iter + 1) % 3 == 0:
            set_requires_grad(self.discriminator, True)
            self.optimizer_discriminator.zero_grad()
            self.backward_discriminator()
            self.optimizer_discriminator.step()

        # update GANLoss
        set_requires_grad(self.discriminator, False)
        self.optimizer_generator.zero_grad()
        self.backward_generator()
        self.optimizer_generator.step()

    def backward_generator(self):
        fake_ab = torch.cat((self.input_a, self.fake_b), 1)
        pred_fake = self.discriminator(fake_ab.detach())
        self.loss_G = self.criterionBCE(pred_fake, torch.ones_like(pred_fake))
        self.loss_vgg = self.criterionVGG(self.fake_b.detach(), self.input_b).to(self.device)
        self.loss_mask = self.criterionL1(self.fake_mask, self.real_mask)
        self.loss_w = self.criterionL1(self.fake_w * self.real_mask, self.real_w * self.real_mask)
        self.loss_alpha = self.criterionL1(self.fake_alpha * self.real_mask, self.real_alpha * self.real_mask)
        self.loss_w_i1 = self.criterionL1(self.fake_i_w * self.real_mask, self.input_b * self.real_mask)
        self.loss_w_i2 = self.criterionL1(self.fake_b * self.real_mask, self.input_b * self.real_mask)
        self.loss_G_all = self.loss_G + self.lambdaVgg * self.loss_vgg + self.lambdaMask * self.loss_mask + \
                          self.lambdaW * self.loss_w + self.lambdaAlpha * self.loss_alpha + \
                          self.lambdaI1 * self.loss_w_i1 + self.lambdaI2 * self.loss_w_i2
        self.loss_G_all.backward()

    def backward_discriminator(self):
        # Global Discriminator
        fake_ab = torch.cat((self.input_a, self.fake_b), 1)
        pred_fake = self.discriminator(fake_ab)

        real_ab = torch.cat((self.input_a, self.input_b), 1)
        pred_real = self.discriminator(real_ab)

        self.loss_D_fake = self.criterionBCE(pred_fake.detach(), torch.zeros_like(pred_fake))
        self.loss_D_real = self.criterionBCE(pred_real, torch.ones_like(pred_real))
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def get_current_visuals(self):
        input_image = self.input_a.data.cpu()[0, :, :, :]
        b, c, h, w = self.input_a.size()
        fake_b = self.fake_b.data.cpu()[0, :, :, :]
        real_b = self.input_b.data.cpu()[0, :, :, :]
        fake_mask = self.fake_mask.data.cpu()[0, :, :, :].expand(3, h, w)
        fake_alpha = self.fake_alpha.data.cpu()[0, :, :, :].expand(3, h, w)
        fake_w = self.fake_w.data.cpu()[0, :, :, :].expand(3, h, w)
        real_mask = self.real_mask.data.cpu()[0, :, :, :].expand(3, h, w)
        real_alpha = self.real_alpha.data.cpu()[0, :, :, :].expand(3, h, w)
        real_w = self.real_w.data.cpu()[0, :, :, :].expand(3, h, w)
        return input_image, fake_b, real_b, \
               fake_mask, fake_alpha, fake_w, \
               real_mask, real_alpha, real_w

    def get_current_errors(self):
        # show the current loss
        data = OrderedDict([('loss_G_ALL', self.loss_G_all.data),
                            ('loss_mask', self.loss_mask.data),
                            ('loss_w', self.loss_w.data),
                            ('loss_alpha', self.loss_alpha.data),
                            ('loss_w_i1', self.loss_w_i1.data),
                            ('loss_w_i2', self.loss_w_i2.data),
                            ])
        if (self.iter + 1) % 3 == 0:
            data.update({
                "loss_D": self.loss_D.data
            })
        return data

    @torch.no_grad()
    def inference(self, image):
        if len(image.size()) == 3:
            image_tensor = torch.unsqueeze(image.to(self.device), dim=0)
        else:
            image_tensor = image.to(self.device)
        fake_b, fake_mask, fake_alpha, fake_w, fake_i_w = self.generator(image_tensor)
        return fake_b, fake_mask, fake_alpha

    def set_mode(self, mode=None):
        if mode is None:
            mode = self.mode
        if mode == "eval":
            self.generator.eval()
            if self.mode == "train":
                self.discriminator.eval()
        else:
            self.generator.train()
            if self.mode == "train":
                self.discriminator.train()
