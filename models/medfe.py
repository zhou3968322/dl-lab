# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/14
import networks
import os
import torch
from utils.dl_util import init_net
from loss.gan_loss import VGG16, PerceptualLoss, StyleLoss, GANLoss
from utils.dl_util import get_scheduler, get_optimizer, set_requires_grad, tensor2im
from models.base_model import BaseModel
from collections import OrderedDict


class Medfe(BaseModel):

    def __init__(self, config):
        super(Medfe, self).__init__(config)
        arch_config = config.pop('arch')
        init_args = arch_config.pop("init_args")
        self.device0 = torch.device("cuda:0")
        self.device1 = torch.device("cuda:1")
        model_names = []
        self.encoder = getattr(getattr(networks, arch_config["encoder"].pop("type")),
                               arch_config["encoder"].pop("name"))(**arch_config["encoder"])
        self.encoder = init_net(self.encoder, **init_args)
        model_names.append("encoder")
        self.decoder = getattr(getattr(networks, arch_config["decoder"].pop("type")),
                               arch_config["decoder"].pop("name"))(**arch_config["decoder"])
        self.decoder = init_net(self.decoder, **init_args)
        model_names.append("decoder")
        if "mask_encoder" in arch_config:
            self.has_mask_encoder = True
            self.mask_encoder = getattr(getattr(networks, arch_config["mask_encoder"].pop("type")),
                                        arch_config["mask_encoder"].pop("name"))(**arch_config["mask_encoder"])
            self.mask_encoder = init_net(self.mask_encoder, **init_args)
            model_names.append("mask_encoder")
        else:
            self.has_mask_encoder = False
        self.mask_decoder = getattr(getattr(networks, arch_config["mask_decoder"].pop("type")),
                                    arch_config["mask_decoder"].pop("name"))(**arch_config["mask_decoder"])
        self.mask_decoder = init_net(self.mask_decoder, **init_args)
        self.mask_decoder.reset_device(self.device0)
        model_names.append("mask_decoder")
        user_inner_loss = arch_config.get("user_inner_loss", False)
        self.pc_block = getattr(getattr(networks, arch_config["pc_block"].pop("type")),
                                arch_config["pc_block"].pop("name"))(**arch_config["pc_block"],
                                                                     use_inner_loss=user_inner_loss)
        self.use_inner_loss = self.pc_block.use_inner_loss
        self.use_fake_mask = arch_config.get("use_fake_mask", False)
        self.pc_block = init_net(self.pc_block, **init_args, gpu_ids=[1])
        self.pc_block.reset_device(self.device1)
        model_names.append("pc_block")
        if self.mode == "train":
            self.discriminator_gt = getattr(getattr(networks, arch_config["discriminator_gt"].pop("type")),
                                            arch_config["discriminator_gt"].pop("name"))(
                **arch_config["discriminator_gt"])
            self.discriminator_gt = init_net(self.discriminator_gt, **init_args)
            model_names.append("discriminator_gt")
            self.discriminator_mask = getattr(getattr(networks, arch_config["discriminator_mask"].pop("type")),
                                              arch_config["discriminator_mask"].pop("name"))(
                **arch_config["discriminator_mask"])
            self.discriminator_mask = init_net(self.discriminator_mask, **init_args)
            model_names.append("discriminator_mask")

        self.model_names = model_names

        if self.mode == "train":
            self.vgg = VGG16().to(self.device0)
            # if len(self.gpu_ids) > 1:
            #     self.vgg = torch.nn.DataParallel(self.vgg, self.gpu_ids)
            loss_args = config.pop("loss")
            self.PerceptualLoss = PerceptualLoss(vgg_module=self.vgg)
            self.lambdaP = loss_args.pop("lambdaP")
            self.StyleLoss = StyleLoss(vgg_module=self.vgg)
            self.lambdaS = loss_args.pop("lambdaS")
            self.criterionL1 = torch.nn.L1Loss()
            self.lambdaL1 = loss_args.pop("lambdaL1")
            if len(self.gpu_ids) > 0:
                self.criterionGAN = GANLoss(tensor=torch.cuda.FloatTensor)
            else:
                self.criterionGAN = GANLoss()
            self.lambdaGan = loss_args.pop("lambdaGan")

        # self.reset_device()

        if self.mode == "train":
            optimizer_args = config["trainer"].pop("optimizer")
            scheduler_args = config["trainer"].pop("scheduler")
            schedulers = []
            self.optimizer_encoder = get_optimizer(self.encoder, **optimizer_args)
            self.optimizer_decoder = get_optimizer(self.decoder, **optimizer_args)
            self.optimizer_mask_decoder = get_optimizer(self.mask_decoder, **optimizer_args)
            self.optimizer_pc_block = get_optimizer(self.pc_block, **optimizer_args)
            self.optimizer_discriminator_gt = get_optimizer(self.discriminator_gt, **optimizer_args)
            self.optimizer_discriminator_mask = get_optimizer(self.discriminator_mask, **optimizer_args)
            optimizers = [self.optimizer_encoder, self.optimizer_decoder, self.optimizer_pc_block,
                          self.optimizer_discriminator_gt, self.optimizer_discriminator_mask]
            if self.has_mask_encoder:
                self.optimizer_mask_encoder = get_optimizer(self.mask_encoder, **optimizer_args)
                optimizers.append(self.optimizer_mask_encoder)
            for optimizer in optimizers:
                schedulers.append(get_scheduler(optimizer, **scheduler_args))
            self.optimizers = optimizers
            self.schedulers = schedulers

        if self.mode == "train" and self.continue_train:
            load_epoch = config["trainer"]["load_epoch"]
            self.load_networks(load_epoch)
        self.print_networks()

    def train(self, data):
        self.input_a = data["A"].to(self.device)
        self.img_name = os.path.basename(data["AB_path"][0]).rsplit('.', 1)[0]
        self.input_b = data["B"].to(self.device)
        self.crop_box = data["crop_box"][0]
        self.noise_mask = (data["noise_mask"].to(self.device))  # noise_mask, hole is 0.0

        fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6 = self.encoder(self.input_a)
        De_in = [fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6]
        if self.has_mask_encoder:
            fake_m_1, fake_m_2, fake_m_3, fake_m_4, fake_m_5, fake_m_6 = self.encoder(self.input_a)
            self.fake_mask = self.mask_decoder(fake_m_1, fake_m_2, fake_m_3, fake_m_4, fake_m_5, fake_m_6)
        else:
            self.fake_mask = self.mask_decoder(fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6)
        if self.use_inner_loss:
            if self.use_fake_mask:
                x_out, loss = self.pc_block(De_in, self.fake_mask)
            else:
                x_out, loss = self.pc_block(De_in, self.noise_mask)
        else:
            if self.use_fake_mask:
                x_out = self.pc_block(De_in, self.fake_mask)
            else:
                x_out = self.pc_block(De_in, self.noise_mask)
        # x_out_real = self.pc_block(De_in, self.fake_mask)
        if self.use_inner_loss:
            self.x_texture_fi = loss[0]
            self.x_structure_fi = loss[1]
            fake_b_1, fake_b_2, fake_b_3, fake_b_4, fake_b_5, fake_b_6 = self.encoder(self.input_b)
            De_b_in = [fake_b_1, fake_b_2, fake_b_3, fake_b_4, fake_b_5, fake_b_6]
            self.x_texture_gt, self.x_structure_gt = self.pc_block.get_features(De_b_in)
        # x_out_real = self.pc_block(De_in, self.fake_mask)
        self.fake_out = self.decoder(x_out[0], x_out[1], x_out[2],
                                     x_out[3], x_out[4], x_out[5])
        # self.fake_out_real = self.decoder(x_out_real[0], x_out_real[1], x_out_real[2],
        #                                   x_out_real[3], x_out_real[4], x_out_real[5])
        self.backward()

    def backward(self):
        # Optimize the D and F first
        set_requires_grad(self.discriminator_gt, True)
        set_requires_grad(self.discriminator_mask, True)
        set_requires_grad(self.encoder, False)
        set_requires_grad(self.decoder, False)
        set_requires_grad(self.mask_decoder, False)
        set_requires_grad(self.pc_block, False)
        self.optimizer_discriminator_mask.zero_grad()
        self.optimizer_discriminator_gt.zero_grad()
        self.backward_discriminator()
        self.optimizer_discriminator_gt.step()
        self.optimizer_discriminator_mask.step()

        # optimize encoder, decoder, pc_block
        set_requires_grad(self.discriminator_gt, False)
        set_requires_grad(self.discriminator_mask, False)
        set_requires_grad(self.encoder, True)
        set_requires_grad(self.decoder, True)
        set_requires_grad(self.mask_decoder, True)
        set_requires_grad(self.pc_block, True)
        self.optimizer_encoder.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_mask_decoder.zero_grad()
        self.pc_block.zero_grad()
        self.backward_generator()
        self.optimizer_pc_block.step()
        self.optimizer_mask_decoder.step()
        self.optimizer_decoder.step()
        self.optimizer_encoder.step()

    def backward_generator(self):
        # First, The generator should fake the discriminator
        # fake_local = self.fake_out[:, :, self.crop_x:self.crop_x + 64, self.crop_y:self.crop_y + 64]
        # Global discriminator
        pred_real = self.discriminator_gt(self.input_b)
        pred_fake = self.discriminator_gt(self.fake_out.detach())

        pred_real_mask = self.discriminator_mask(self.noise_mask)
        pred_fake_mask = self.discriminator_mask(self.fake_mask.detach())

        # Local discriminator
        input_a_local = self.input_a[:, :, self.crop_box[0]: self.crop_box[2], self.crop_box[1]: self.crop_box[3]]
        input_b_local = self.input_b[:, :, self.crop_box[0]: self.crop_box[2], self.crop_box[1]: self.crop_box[3]]
        fake_out_local = self.fake_out[:, :, self.crop_box[0]: self.crop_box[2],
                         self.crop_box[1]: self.crop_box[3]].detach()
        noise_mask_local = self.noise_mask[:, :, self.crop_box[0]: self.crop_box[2], self.crop_box[1]: self.crop_box[3]]
        fake_mask_local = self.fake_mask[:, :, self.crop_box[0]: self.crop_box[2],
                          self.crop_box[1]: self.crop_box[3]].detach()

        pred_real_local = self.discriminator_gt(input_a_local)
        pred_fake_local = self.discriminator_gt(fake_out_local)
        pred_real_mask_local = self.discriminator_mask(noise_mask_local)
        pred_fake_mask_local = self.discriminator_mask(fake_mask_local)

        # whole image loss
        if self.use_inner_loss:
            self.feature_loss = self.criterionL1(self.x_texture_fi, self.x_structure_fi) + \
                                self.criterionL1(self.x_structure_fi, self.x_structure_gt)
        self.loss_G_GAN = self.criterionGAN(pred_fake, pred_real, False) + \
                          self.criterionGAN(pred_fake_local, pred_real_local, False)
        self.loss_l1 = self.criterionL1(self.fake_out, self.input_b) + \
                       self.criterionL1(fake_out_local, input_b_local)
        self.perceptual_loss = self.PerceptualLoss(self.fake_out, self.input_b)
        self.style_loss = self.StyleLoss(self.fake_out, self.input_b)

        self.loss_G_GAN_mask = self.criterionGAN(pred_fake_mask, pred_real_mask, False) + \
                               self.criterionGAN(pred_fake_mask_local, pred_real_mask_local, False)
        self.loss_l1_mask = self.criterionL1(self.fake_mask, self.noise_mask) + \
                            self.criterionL1(fake_mask_local, noise_mask_local)

        # 修改为
        # self.loss_G = self.loss_G_L1 * 20 + self.loss_G_GAN * 1 + self.Perceptual_loss * 1 + self.Style_Loss * 2
        # self.loss_G = self.loss_G_L1 + self.loss_G_GAN *0.2 + self.Perceptual_loss * 0.2 + self.Style_Loss *250
        self.loss_G = self.loss_l1 * self.lambdaL1 + self.loss_G_GAN * self.lambdaGan + \
                      self.perceptual_loss * self.lambdaP + self.style_loss * self.lambdaS
        self.loss_G_mask = self.loss_l1_mask * self.lambdaL1 / 2 + self.loss_G_GAN_mask * self.lambdaGan / 2
        if self.use_inner_loss:
            self.feature_loss.backward(retain_graph=True)
        if self.use_fake_mask or not self.has_mask_encoder:
            self.loss_G.backward(retain_graph=True)
        else:
            self.loss_G.backward()
        self.loss_G_mask.backward()

    def backward_discriminator(self):
        # Global Discriminator
        self.pred_fake = self.discriminator_gt(self.fake_out.detach())  # D(G(z))
        self.pred_real = self.discriminator_gt(self.input_b)  # D(x)

        self.pred_fake_mask = self.discriminator_mask(self.fake_mask.detach())
        self.pred_real_mask = self.discriminator_mask(self.noise_mask)

        # Local discriminator
        input_a_local = self.input_a[:, :, self.crop_box[0]: self.crop_box[2], self.crop_box[1]: self.crop_box[3]]
        fake_out_local = self.fake_out[:, :, self.crop_box[0]: self.crop_box[2],
                         self.crop_box[1]: self.crop_box[3]].detach()
        noise_mask_local = self.noise_mask[:, :, self.crop_box[0]: self.crop_box[2], self.crop_box[1]: self.crop_box[3]]
        fake_mask_local = self.fake_mask[:, :, self.crop_box[0]: self.crop_box[2],
                          self.crop_box[1]: self.crop_box[3]].detach()

        pred_real_local = self.discriminator_gt(input_a_local)
        pred_fake_local = self.discriminator_gt(fake_out_local)
        pred_real_mask_local = self.discriminator_mask(noise_mask_local)
        pred_fake_mask_local = self.discriminator_mask(fake_mask_local)

        self.loss_D = self.criterionGAN(self.pred_fake, self.pred_real, True) + \
                      self.criterionGAN(pred_fake_local, pred_real_local, True)
        self.loss_D_mask = self.criterionGAN(self.pred_fake_mask, self.pred_real_mask, True) + \
                           self.criterionGAN(pred_fake_mask_local, pred_real_mask_local, True)
        self.loss_D.backward()
        self.loss_D_mask.backward()

    def get_current_visuals(self):
        input_image = (self.input_a.data.cpu()[0, :, :, :] + 1) / 2.0
        b, c, h, w = self.noise_mask.size()
        fake_mask = ((self.fake_mask.data.cpu()[0, :, :, :] + 1) / 2.0).expand(3, h, w)
        noise_mask = ((self.noise_mask.data.cpu()[0, :, :, :] + 1) / 2.0).expand(3, h, w)
        # fake_image_real = (self.fake_out_real.data.cpu() + 1) / 2.0
        fake_image = (self.fake_out.data.cpu()[0, :, :, :] + 1) / 2.0
        real_gt = (self.input_b.data.cpu()[0, :, :, :] + 1) / 2.0
        return input_image, fake_mask, noise_mask, fake_image, real_gt
        # return input_image, fake_mask, noise_mask, fake_image_real, fake_image, real_gt

    def get_current_errors(self):
        # show the current loss
        return OrderedDict([('G_GAN', self.loss_G_GAN.data),
                            ('G_GAN_mask', self.loss_G_GAN_mask.data),
                            ('D_Loss', self.loss_D.data),
                            ('D_mask_Loss', self.loss_D_mask.data),
                            ('G_L1', self.loss_l1.data),
                            ('G_L1_mask', self.loss_l1_mask.data),
                            ('PerceptualLoss', self.perceptual_loss.data),
                            ('StyleLoss', self.style_loss.data)
                            ])
