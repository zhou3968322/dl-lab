# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/1/12
import networks
import os
import torch
from utils.dl_util import init_net
from loss.gan_loss import ConditionGANLoss
from utils.dl_util import get_scheduler, get_optimizer, set_requires_grad, tensor2im
from models.base_model import BaseModel
from collections import OrderedDict
from metrics import metrics


class Rascv2(BaseModel):

    def __init__(self, config):
        super(Rascv2, self).__init__(config)
        arch_config = config.pop('arch')
        init_args = arch_config.pop("init_args")
        model_names = []
        self.generator = getattr(getattr(networks, arch_config["generator"].pop("type")),
                                 arch_config["generator"].pop("name"))(**arch_config["generator"])
        self.generator = init_net(self.generator, **init_args)
        model_names.append("generator")

        if self.mode == "train":
            self.discriminator = getattr(getattr(networks, arch_config["discriminator"].pop("type")),
                                         arch_config["discriminator"].pop("name"))(
                **arch_config["discriminator"])
            self.discriminator = init_net(self.discriminator, **init_args)
            model_names.append("discriminator")
        self.model_names = model_names

        if self.mode == "train":
            # if len(self.gpu_ids) > 1:
            #     self.vgg = torch.nn.DataParallel(self.vgg, self.gpu_ids)
            loss_args = config.pop("loss")
            gan_mode = loss_args.get("gan_mode", "lsgan")
            self.criterionL1 = torch.nn.L1Loss()
            self.lambdaL1 = loss_args.pop("lambdaL1")
            self.criterionGAN = ConditionGANLoss(gan_mode=gan_mode)

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

    def train(self, data):
        self.input_a = data["A"].to(self.device)
        self.img_name = os.path.basename(data["AB_path"][0]).rsplit('.', 1)[0]
        self.input_b = data["B"].to(self.device)
        self.fake_b = self.generator(self.input_a)
        self.backward()

    def backward(self):
        # update D
        set_requires_grad(self.discriminator, True)
        self.optimizer_discriminator.zero_grad()
        self.backward_discriminator()
        self.optimizer_discriminator.step()

        # update GANLoss
        set_requires_grad(self.discriminator, False)
        self.optimizer_generator.zero_grad()
        self.backward_generator()
        self.optimizer_generator.step()

    @torch.no_grad()
    def inference(self, image):
        if len(image.size()) == 3:
            image_tensor = torch.unsqueeze(image.to(self.device), dim=0)
        else:
            image_tensor = image.to(self.device)
        fake_b = self.generator(image_tensor)
        return fake_b

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

    def backward_generator(self):
        fake_ab = torch.cat((self.input_a, self.fake_b), 1)
        pred_fake = self.discriminator(fake_ab.detach())
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_b, self.input_b) * self.lambdaL1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def backward_discriminator(self):
        # Global Discriminator
        fake_ab = torch.cat((self.input_a, self.fake_b), 1)
        pred_fake = self.discriminator(fake_ab.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_ab = torch.cat((self.input_a, self.input_b), 1)
        pred_real = self.discriminator(real_ab)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def get_current_visuals(self):
        # mask is 0, 1 ,image is  -1, 1
        input_image = (self.input_a.data.cpu()[0, :, :, :] + 1) / 2.0
        b, c, h, w = self.input_a.size()
        fake_b = ((self.fake_b.data.cpu()[0, :, :, :] + 1) / 2.0).expand(3, h, w)
        real_b = ((self.input_b.data.cpu()[0, :, :, :] + 1) / 2.0).expand(3, h, w)
        return input_image, fake_b, real_b
        # return input_image, fake_mask, noise_mask, fake_image_real, fake_image, real_gt

    def get_evaluate_errors(self):
        mse = metrics.MSE()(self.fake_b.detach(), self.input_b)
        psnr = metrics.PSNR()(self.fake_b.detach(), self.input_b)
        ssim = metrics.SSIM()(self.fake_b.detach(), self.input_b)
        data = OrderedDict([('mse', mse.data),
                            ('psnr', psnr.data),
                            ("ssim", ssim),
                            ])
        return data

    def get_current_errors(self):
        # show the current loss
        data = OrderedDict([('G_GAN', self.loss_G_GAN.data),
                            ('D_fake', self.loss_D_fake.data),
                            ('D_real', self.loss_D_real.data),
                            ('D_Loss', self.loss_D.data),
                            ('G_L1', self.loss_G_L1.data),
                            ])
        return data
