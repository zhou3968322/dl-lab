# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/3
import lmdb
from torch.utils.data.dataset import Dataset
from utils.log import logger


class CrnnLmdbDataset(Dataset):

    def __init__(self, lmdb_dir):
        super(CrnnLmdbDataset, self).__init__()
        n_samples = 0
        envs = {}
        for _ in lmdb_dir:
            _env = lmdb.open(_, readonly=True, max_dbs=3, lock=False, readahead=False)
            _n_samples_db = _env.open_db(b"n_samples")
            with _env.begin(write=False) as txn:
                n_samples_cursor = txn.cursor(_n_samples_db)
                n_sample = int(n_samples_cursor.get(b"n_samples"))
                if n_sample == 0:
                    logger.warning("there is no train data in {}".format(_))
                    continue
                n_samples += n_sample
            envs.update({n_samples: _env})
        self._envs = envs
