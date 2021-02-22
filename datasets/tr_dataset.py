# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2020/12/3
from torch.utils.data import Dataset
import lmdb
import six
import sys
from PIL import Image
from utils.log import logger

from_ch = [
    u'，', u'！', u'：', u'（', u'）', u'；', u'—', u'“', u'”', u'‘', u'’', u'～', u'√', u'℃', u'￥', u'в', u'［', u'］', u'｜',
    u'•', u'─', u'－', u'—', u'–', u'-', u'\n', u'／', u'？', u'＜', u'＞', u"〖", u'〗', u'〇', u'…', u'°', u'′'
]
to_ch = [
    u',', u'!', u':', u'(', u')', u';', u'-', u'"', u'"', u'\'', u'\'', u'~', u'V', u'C', u'¥', u'B', u'[', u']', u'|',
    u'·', u'-', u'-', u'-', u'-', u'-', u'', u'/', u'?', u'<', u'>', u"[", u']', u'O', u'...', u'。', u'\''
]


def replace_character(text):
    result_text = text
    for i, c in enumerate(from_ch):
        if c in result_text:
            result_text = result_text.replace(c, to_ch[i])

    return result_text


class TrDataset(Dataset):
    '''采用lmdb工具进行数据读取
    '''

    def __init__(self, data_root=None, replace_ch=True, transform=None, target_transform=None,
                 max_text_len=40, data_filtering_off=True):
        super(TrDataset, self).__init__()
        self.data_root = data_root
        self.env = lmdb.open(
            data_root,
            max_dbs=3,
            readonly=True,
            lock=False,
            readahead=False)
        if not self.env:
            logger.error('cannot creat lmdb from %s' % (data_root))
            sys.exit(0)
        _n_samples_db = self.env.open_db(b"n_samples")
        n_samples = 0
        with self.env.begin(write=False) as txn:
            n_samples_cursor = txn.cursor(_n_samples_db)
            n_sample = int(n_samples_cursor.get(b"n_samples"))
            n_samples += n_sample
            if n_sample == 0:
                logger.warning("there is no train data in {}".format(_))
        self.n_samples = n_sample
        self.transform = transform
        self.target_transform = target_transform
        self.replace_ch = replace_ch
        self.max_text_len = max_text_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        _image_db = self.env.open_db(b"image")
        _label_db = self.env.open_db(b"label")
        with self.env.begin(write=False) as txn:
            image_cursor = txn.cursor(db=_image_db)
            label_cursor = txn.cursor(db=_label_db)
            byte_index = bytes('%09d' % (index + 1), encoding='utf-8')
            imgbuf = image_cursor.get(byte_index)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf)  # brightness 调整的时候需要是彩色图像
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            label = str(label_cursor.get(byte_index), encoding='utf-8')
            if self.replace_ch:
                label = replace_character(label)
            if len(label) > self.max_text_len:
                print("text too long with label:{}".format(label))
            if self.target_transform is not None:
                label = self.target_transform(label)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return {"img": img, "label": label}


class TrResearchDataset(Dataset):
    '''采用lmdb工具进行数据读取
    '''

    def __init__(self, data_root=None, replace_ch=True, transform=None, target_transform=None,
                 max_text_len=40, data_filtering_off=False):
        super(TrResearchDataset, self).__init__()
        self.data_root = data_root
        self.env = lmdb.open(data_root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            logger.error('cannot creat lmdb from %s' % (data_root))
            sys.exit(0)
        _n_samples_db = self.env.open_db(b"n_samples")
        n_samples = 0
        with self.env.begin(write=False) as txn:
            n_samples_cursor = txn.cursor(_n_samples_db)
            n_sample = int(n_samples_cursor.get(b"n_samples"))
            n_samples += n_sample
            if n_sample == 0:
                logger.warning("there is no train data in {}".format(_))
            if data_filtering_off:
                # for fast check with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                # Filtering
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > self.opt.batch_max_length or len(label) == 0:
                        # print(f'The length of the label is longer than max_length: length \
                        #     {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)
        self.n_samples = n_sample
        self.transform = transform
        self.target_transform = target_transform
        self.replace_ch = replace_ch
        self.max_text_len = max_text_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        _image_db = self.env.open_db(b"image")
        _label_db = self.env.open_db(b"label")
        with self.env.begin(write=False) as txn:
            image_cursor = txn.cursor(db=_image_db)
            label_cursor = txn.cursor(db=_label_db)
            byte_index = bytes('%09d' % (index + 1), encoding='utf-8')
            imgbuf = image_cursor.get(byte_index)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf)  # brightness 调整的时候需要是彩色图像
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            label = str(label_cursor.get(byte_index), encoding='utf-8')
            if self.replace_ch:
                label = replace_character(label)
            if len(label) > self.max_text_len:
                print("text too long with label:{}".format(label))
            if self.target_transform is not None:
                label = self.target_transform(label)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return {"img": img, "label": label}
