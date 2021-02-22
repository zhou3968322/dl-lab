# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/1/29

from utils.common_util import get_model_name
import models, datasets
from utils.log import logger
import time, torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.collate_fn import PaddingCollate
from utils.averager import Averager
from nltk.metrics.distance import edit_distance


class SrnTrainer(object):

    def __init__(self, config):
        self.experiment_name = config.pop('name')
        self.random_seed = config.get('random_seed', 30)

        self.start_epoch = config["trainer"]["scheduler"]["start_epoch"]
        self.niter = config["trainer"]["scheduler"]["niter"]
        self.niter_decay = config["trainer"]["scheduler"]["niter_decay"]

        train_dataset_args = config["datasets"]["train"]["dataset"]
        train_dataset_type = train_dataset_args["type"]
        train_dataset_init_args = train_dataset_args["args"]
        train_dataset = getattr(datasets, train_dataset_type)(**train_dataset_init_args)
        train_data_loader_args = config["datasets"]["train"]["loader"]
        collate_fn = PaddingCollate(imgH=config["arch"]["srn"].get("img_h", 48),
                                    imgW=config["arch"]["srn"].get("img_w", 800),
                                    nc=config["arch"]["srn"].get("input_channel", 1))
        logger.info("train dataset len:{}".format(len(train_dataset)))
        train_loader = DataLoader(dataset=train_dataset, collate_fn=collate_fn, **train_data_loader_args)
        self.train_loader = train_loader

        val_dataset_args = config["datasets"]["val"]["dataset"]
        val_dataset_type = val_dataset_args["type"]
        val_dataset_init_args = val_dataset_args["args"]
        val_dataset = getattr(datasets, val_dataset_type)(**val_dataset_init_args)
        val_data_loader_args = config["datasets"]["val"]["loader"]
        logger.info("val dataset len:{}".format(len(val_dataset)))
        val_collate_fn = PaddingCollate(imgH=config["arch"]["srn"].get("img_h", 48),
                                        imgW=config["arch"]["srn"].get("img_w", 800),
                                        nc=config["arch"]["srn"].get("input_channel", 1))
        val_loader = DataLoader(dataset=val_dataset, collate_fn=val_collate_fn, **val_data_loader_args)
        self.val_loader = val_loader

        model_name = get_model_name(config["arch"].pop("type"))
        self.model = getattr(models, model_name)(config)
        logger.info("model init success")

        tensorboard_log_dir = config["trainer"]["log_dir"]
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir, comment=self.experiment_name)
        self.display_val_freq = config["trainer"]["display_val_freq"]
        # self.evaluate_freq = config["trainer"]["evaluate_freq"]
        self.print_freq = config["trainer"]["print_freq"]
        self.evaluate_freq = config["trainer"]["evaluate_freq"]
        self.save_epoch_freq = config["trainer"].get("save_epoch_freq", 0)
        self.save_step_freq = config["trainer"].get("save_step_freq", 0)
        self.model.set_mode()

        self.global_step = 0

    def train(self):
        for epoch in range(self.start_epoch, self.niter + self.niter_decay + 1):
            self.model.loss_avg.reset()
            self.train_one_epoch(epoch)
        self.writer.close()

    def validate(self):
        valid_loss_avg = Averager()
        n_correct = 0
        norm_ed = 0
        length_of_data = 0
        not_sup_data_len = 0
        val_iter = 0
        for val_data in self.val_loader:
            batch_size = val_data["images"].size(0)
            is_valid = self.model.check_data(val_data)
            if is_valid:
                val_iter += 1
                length_of_data += batch_size
                cost, preds_str, labels = self.model.evaluate(val_data)
                valid_loss_avg.add(cost)
                for pred, gt in zip(preds_str, labels):
                    if pred.strip() == gt.strip():
                        n_correct += 1
                    else:
                        if len(gt) == 0:
                            norm_ed += 1
                        else:
                            norm_ed += edit_distance(pred, gt) / len(gt)
                if val_iter % self.display_val_freq == 0:
                    grid = torchvision.utils.make_grid(val_data["images"], nrow=4)
                    label_text = "\n".join(val_data["labels"])
                    pred_text = "\n".join(preds_str)
                    self.writer.add_image('val_show_img_{}'.format(val_iter), grid, val_iter)
                    self.writer.add_text("val_show_gt_text_{}".format(val_iter), label_text, val_iter)
                    self.writer.add_text("val_show_pred_text_{}".format(val_iter), pred_text, val_iter)
            else:
                not_sup_data_len += 1
                label_text = "\n".join(val_data["labels"])
                grid = torchvision.utils.make_grid(val_data["images"], nrow=4)
                self.writer.add_image('val_too_long_img_{}'.format(not_sup_data_len), grid)
                self.writer.add_text("val_too_long_gt_text_{}".format(not_sup_data_len), label_text)

        accuracy = n_correct / float(length_of_data) * 100
        return valid_loss_avg.val(), accuracy, norm_ed, length_of_data

    def train_one_epoch(self, epoch):
        epoch_start_time = time.time()
        for train_data in self.train_loader:
            iter_start_time = time.time()
            self.model.train(train_data)
            self.global_step += 1
            if self.global_step % self.print_freq == 0:
                errors = self.model.get_current_errors()
                t_comp = time.time() - iter_start_time
                message = 'experiment:%s, train, (epoch: %d, steps: %d, time: %.3f) ' % (self.experiment_name, epoch,
                                                                                         self.global_step, t_comp)
                for key, value in errors.items():
                    message += '%s: %.5f ' % (key, value)
                    self.writer.add_scalar("train/{}".format(key), errors[key], self.global_step)
                logger.info(message)
            if self.global_step % self.evaluate_freq == 0:
                self.model.set_mode(mode="eval")
                valid_start = time.time()
                valid_error, accuracy, norm_ed, length_of_data = self.validate()
                valid_cost = time.time() - valid_start
                self.model.set_mode(mode="train")
                message = 'experiment:%s, val, (epoch: %d, steps: %d, time: %.3f) ' % (self.experiment_name, epoch,
                                                                                       self.global_step, valid_cost)
                message += 'loss_val: %.5f ' % valid_error
                message += 'accuracy: %.5f ' % accuracy
                message += "norm_ed: %.5f" % norm_ed
                message += "data_len: %d" % length_of_data
                self.writer.add_scalar("val/loss", valid_cost, self.global_step)
                logger.info(message)

            # if self.global_step % self.display_freq == 0:
            #     visual_input = self.model.get_current_visuals()
            #     grid = torchvision.utils.make_grid(list(visual_input), nrow=3)
            #     img_name = self.model.img_name
            #     self.writer.add_image('experiment_{}_train_epoch_{}_step_{}_img_name_{}'.
            #                           format(self.experiment_name, epoch, self.global_step, img_name), grid,
            #                           self.global_step)
            # if self.global_step % self.evaluate_freq == 0:
            #     self.model.set_mode(mode="eval")
            #     fake_mask, fake_out = self.model.inference(train_data["A"])
            #     fake_out_test = self.model.inference(train_data["A"], train_data["noise_mask"])
            #     b, c, h, w = fake_mask.size()
            #     input_image = (train_data["A"].data.cpu()[0, :, :, :] + 1) / 2.0
            #     fake_mask = ((fake_mask.data.cpu()[0, :, :, :] + 1) / 2.0).expand(3, h, w)
            #     noise_mask = ((train_data["noise_mask"].data.cpu()[0, :, :, :] + 1) / 2.0).expand(3, h, w)
            #     fake_image = (fake_out.data.cpu()[0, :, :, :] + 1) / 2.0
            #     fake_image_test = ((fake_out_test.data.cpu()[0, :, :, :] + 1) / 2.0).expand(3, h, w)
            #     real_gt = (train_data["B"].data.cpu()[0, :, :, :] + 1) / 2.0
            #     visuals = [input_image, fake_mask, noise_mask, fake_image, fake_image_test, real_gt]
            #     grid = torchvision.utils.make_grid(visuals, nrow=3)
            #     img_name = self.model.img_name
            #     self.writer.add_image('experiment_{}_eval_epoch_{}_step_{}_img_name_{}'.
            #                           format(self.experiment_name, epoch, self.global_step, img_name), grid,
            #                           self.global_step + 1)
            #     self.model.set_mode()
            if self.save_epoch_freq == 0 and self.save_step_freq > 0 and self.global_step % self.save_step_freq == 0:
                logger.info('saving the model epoch:{}, step:{}'.format(epoch, self.global_step))
                self.model.save_networks(epoch)
        if self.save_epoch_freq > 0 and epoch % self.save_epoch_freq == 0:
            logger.info('saving the model at the end of epoch:{}'.format(epoch))
            self.model.save_networks(epoch)
        logger.info('End of epoch {} / {} \t Time Taken: {} sec'.format(epoch, self.niter + self.niter_decay,
                                                                        time.time() - epoch_start_time))
        self.model.update_learning_rate()
