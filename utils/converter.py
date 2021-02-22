# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/2/18
import torch
from utils.log import logger


def _load_charsets(charset_path):
    with open(charset_path, 'r') as f:
        charsets = f.read().strip('\n')
    return charsets


class SRNConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, charset_path=None, character=None, batch_max_length=40):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        # list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        if charset_path is not None:
            character = _load_charsets(charset_path)
        assert character is not None
        list_character = list(character)
        self.beg_str = "sos"
        self.end_str = "eos"
        self.character = self.add_special_char(list_character)
        self.PAD = len(self.character) - 1
        self.batch_max_length = batch_max_length
        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def add_special_char(self, dict_character):
        dict_character = dict_character
        # dict_character = dict_character + [self.beg_str] + [self.end_str]
        dict_character = dict_character + [self.end_str]
        return dict_character

    def get_ignored_tokens(self):
        # beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        # return [beg_idx, end_idx]
        return end_idx

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = int(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = int(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx

    def encode(self, text):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """

        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.PAD)
        # mask_text = torch.cuda.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            t = self.clean_text(t)
            t_text = [self.dict[char] for char in t]
            # t_mask = [1 for i in range(len(text) + 1)]
            batch_text[i][0:len(t)] = torch.LongTensor(t_text)  # batch_text[:, len_text+1] = [EOS] token
            # mask_text[i][0:len(text)+1] = torch.cuda.LongTensor(t_mask)
        return batch_text, torch.IntTensor(length)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        text_list = []
        ignored_tokens = self.get_ignored_tokens()
        [beg_idx, end_idx] = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(end_idx):
                    break
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
            text = ''.join(char_list)
            text_list.append(text)
        return text_list

    def clean_text(self, text):
        """
        filter char where not in alphabet with ' '
        """
        clean_txt = ''
        for char in text:
            if char in self.dict:
                clean_txt += char
            else:
                logger.warning(f"alphabet has no {char}")
                clean_txt += ' '
        return clean_txt
