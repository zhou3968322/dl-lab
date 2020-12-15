# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/14
import networks
import torch
from utils.dl_util import init_net
from loss.gan_loss import VGG16, PerceptualLoss, StyleLoss, GANLoss
from utils.dl_util import get_scheduler, get_optimizer, set_requires_grad
from models.base_model import BaseModel


class Medfe(BaseModel):

    def __init__(self, config):
        super(Medfe, self).__init__(config)
        arch_config = config.pop('arch')
        encoder_init_args = arch_config["encoder"].pop("init_args")
        model_names = []
        self.encoder = getattr(getattr(networks, arch_config["encoder"].pop("type")),
                               arch_config["encoder"].pop("name"))(**arch_config["encoder"])
        self.encoder = init_net(self.encoder, **encoder_init_args)
        model_names.append("encoder")
        decoder_init_args = arch_config["decoder"].pop("init_args")
        self.decoder = getattr(getattr(networks, arch_config["decoder"].pop("type")),
                               arch_config["decoder"].pop("name"))(**arch_config["decoder"])
        self.decoder = init_net(self.decoder, **decoder_init_args)
        model_names.append("decoder")
        pc_block_init_args = arch_config["pc_block"].pop("init_args")
        self.pc_block = getattr(getattr(networks, arch_config["pc_block"].pop("type")),
                                arch_config["pc_block"].pop("name"))(**arch_config["pc_block"])
        self.pc_block = init_net(self.pc_block, **pc_block_init_args)
        model_names.append("pc_block")
        if self.mode == "train":
            discriminator_gt_args = arch_config["discriminator_gt"].pop("init_args")
            self.discriminator_gt = getattr(getattr(networks, arch_config["discriminator_gt"].pop("type")),
                                            arch_config["discriminator_gt"].pop("name"))(**arch_config["discriminator_gt"])
            self.discriminator_gt = init_net(self.discriminator_gt, **discriminator_gt_args)
            model_names.append("discriminator_gt")
            discriminator_mask_args = arch_config["discriminator_mask"].pop("init_args")
            self.discriminator_mask = getattr(getattr(networks, arch_config["discriminator_mask"].pop("type")),
                                              arch_config["discriminator_mask"].pop("name"))(**arch_config["discriminator_mask"])
            self.discriminator_mask = init_net(self.discriminator_mask, **discriminator_mask_args)

        self.model_names = model_names

        if self.mode == "train":
            self.vgg = VGG16()
            loss_args = config.pop("loss")
            self.perceptual_loss = PerceptualLoss(vgg_module=self.vgg)
            self.lambdaP = loss_args.pop("lambdaP")
            self.style_loss = StyleLoss(vgg_module=self.vgg)
            self.lambdaS = loss_args.pop("lambdaS")
            self.l1_loss = torch.nn.L1Loss()
            self.lambdaL1 = loss_args.pop("lambdaL1")
            self.gan_loss = loss_args.pop("lambdaGan")

        if self.mode == "train":
            optimizer_args = config["trainer"].pop("optimizer")
            scheduler_args = config["trainer"].pop("scheduler")
            schedulers = []
            self.optimizer_encoder = get_optimizer(self.encoder, **optimizer_args)
            self.optimizer_decoder = get_optimizer(self.decoder, **optimizer_args)
            self.optimizer_pc_block = get_optimizer(self.pc_block, **optimizer_args)
            self.optimizer_discriminator_gt = get_optimizer(self.discriminator_gt, **optimizer_args)
            optimizers = [self.optimizer_encoder, self.optimizer_decoder, self.optimizer_pc_block,
                          self.optimizer_discriminator_gt]
            for optimizer in optimizers:
                schedulers.append(get_scheduler(optimizer, **scheduler_args))
            self.optimizers = optimizers
            self.schedulers = schedulers

        if self.mode == "train" and self.continue_train:
            load_epoch = config["trainer"]["load_epoch"]
            self.load_networks(load_epoch)
        self.print_networks()

    def train(self, data):
        current_data = data

    def backward(self):
        set_requires_grad(self.discriminator_gt, True)
        set_requires_grad(self.encoder, False)
        set_requires_grad(self.decoder, False)
        set_requires_grad(self.pc_block, False)
        self.optimizer_discriminator_gt.zero_grad()

    def backward_discriminator(self):
        pass
