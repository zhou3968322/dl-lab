# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/14
import torch
from torch.nn import init
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.log import logger
from torch.autograd import Variable


def get_gpu_ids(gpu_ids):
    if isinstance(gpu_ids, str):
        return list(map(int, gpu_ids.split(',')))
    else:
        return gpu_ids


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    logger.info('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=None):
    if gpu_ids is None:
        gpu_ids = [0]
    else:
        gpu_ids = get_gpu_ids(gpu_ids)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        if len(gpu_ids) > 1:
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)
        else:
            net.to(torch.device("cuda:{}".format(gpu_ids[0])))
    init_weights(net, init_type, gain=init_gain)
    return net


def get_scheduler(optimizer, lr_policy="lambda", niter=20, start_epoch=1, niter_decay=100, lr_decay_iters=50):
    if lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + start_epoch - niter) / float(niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


def get_optimizer(model, optimizer_type="adam", lr="0.001", beta1=0.05):
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
    elif optimizer_type == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    elif optimizer_type == "rmsprob":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', optimizer_type)
    return optimizer


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    logger.info(net)
    logger.info('Total number of parameters: %d' % num_params)


def cal_feat_mask(inMask, conv_layers, threshold):
    assert inMask.dim() == 4, "mask must be 4 dimensions"
    assert inMask.size(0) == 1, "the first dimension must be 1 for mask"
    inMask = inMask.float()
    convs = []
    inMask = Variable(inMask, requires_grad = False)
    for id_net in range(conv_layers):
        conv = nn.Conv2d(1,1,4,2,1, bias=False)
        conv.weight.data.fill_(1/16)
        convs.append(conv)
    lnet = nn.Sequential(*convs)
    if inMask.is_cuda:

        lnet = lnet.to(inMask.device)
    output = lnet(inMask)
    output = (output > threshold).float().mul_(1)

    return output


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None
