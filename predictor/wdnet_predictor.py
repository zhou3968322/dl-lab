# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/1/21

# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/1/8

from utils.common_util import get_model_name
import models, datasets
from utils.log import logger
from PIL import Image
import os, glob
import time
from torchvision import transforms
from utils.log import logger
from utils.dl_util import tensor2im
from utils.common_util import get_file_list
from utils.img_util import correct_orientation


class WdnetPredictor(object):

    def __init__(self, config):
        self.experiment_name = config.pop('name')
        self.random_seed = config.get('random_seed', 30)
        model_name = get_model_name(config["arch"].pop("type"))
        self.model = getattr(models, model_name)(config)
        logger.info("model init success")
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.long_side = config["predictor"]["long_side"]
        if "test_img_dir" in config["predictor"]:
            self.batch_flag = True
            test_img_dir = config["predictor"]["test_img_dir"]
            self.input_img_paths = get_file_list(test_img_dir, p_postfix=['.jpg', '.png', ".tif", ".JPG", ".jpeg"])
        else:
            self.batch_flag = False
        out_img_dir = config["predictor"]["out_img_dir"]
        os.makedirs(out_img_dir, exist_ok=True)
        self.out_img_dir = out_img_dir
        self.combine_res = config["predictor"].get("combine_res", True)
        self.model.set_mode()

    def predict(self, img=None):
        if self.batch_flag:
            return self.predict_batch()
        else:
            return self.predict_single(img)

    def predict_single(self, single_img):
        assert single_img is not None
        img = single_img
        input_a = Image.new("RGB", (self.long_side, self.long_side), color=(255, 255, 255))
        w, h = img.size
        ratio = self.long_side / max(w, h)
        nw = int(w * ratio)
        nh = int(h * ratio)
        input_a.paste(img.resize((nw, nh), resample=Image.ANTIALIAS))
        img_tensor = self.transform(input_a)
        fake_b, fake_mask, fake_alpha = self.model.inference(img_tensor)
        fake_b = Image.fromarray(tensor2im(fake_b)).crop(box=(0, 0, nw, nh))
        fake_mask = Image.fromarray(tensor2im(fake_mask)).crop(box=(0, 0, nw, nh))
        fake_alpha = Image.fromarray(tensor2im(fake_alpha)).crop(box=(0, 0, nw, nh))
        return fake_b, fake_mask, fake_alpha

    def predict_batch(self):
        sum_time = 0
        for img_path in self.input_img_paths:
            t0 = time.time()
            correct_orientation(img_path)
            img = Image.open(img_path)
            fake_b, fake_mask, fake_alpha = self.predict_single(img)
            logger.info("current img:{} cost:{}".format(img_path, time.time() - t0))
            sum_time += time.time() - t0
            img_name = os.path.basename(img_path)
            file_pref, file_ext = img_name.rsplit(".", 1)
            nw, nh = fake_b.size
            if self.combine_res:
                new_im = Image.new("RGB", (4 * nw, nh), (255, 255, 255))
                new_im.paste(img.resize((nw, nh), resample=Image.ANTIALIAS), box=(0, 0, nw, nh))
                new_im.paste(fake_b.convert("RGB"), box=(nw, 0, 2 * nw, nh))
                new_im.paste(fake_mask.convert("RGB"), box=(2 * nw, 0, 3 * nw, nh))
                new_im.paste(fake_alpha.convert("RGB"), box=(3 * nw, 0, 4 * nw, nh))
                out_img_path = os.path.join(self.out_img_dir, "{}.{}".format(file_pref, file_ext))
                new_im.save(out_img_path)
            else:
                out_mask_path = os.path.join(self.out_img_dir, "{}_mask.{}".format(file_pref, file_ext))
                fake_mask.save(out_mask_path)

        logger.info("average cost:{}".format(sum_time / len(self.input_img_paths)))


