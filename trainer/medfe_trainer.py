# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/14
from utils.common_util import get_model_name
import models


class MedfeTrainer(object):

    def __init__(self, config):
        self.experiment_name = config.pop('name')
        model_name = get_model_name(config["arch"].pop("type"))
        model = getattr(models, model_name)(config["arch"])

    def train(self):
        pass


