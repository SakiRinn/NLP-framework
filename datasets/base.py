import glob
import os
from abc import ABCMeta, abstractmethod
from typing import OrderedDict
import torch

from torch.utils.data import Dataset
from transformers import BertTokenizer

import utils


class BaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(self, data_dir, tokenizer=None):
        self.data_dir = data_dir
        if tokenizer is None:
            self.tokenizer = self.build_tokenizer(data_dir)
        else:
            self.tokenizer = tokenizer
        self.labels = None

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @staticmethod
    def build_tokenizer(data_dir):
        text = ''
        for file in glob.glob(data_dir):
            raise NotImplementedError

        excluded_symbols = "#…!\"$%&'()*+,-—–./:;<=>?@[\\]^_`{|}~‘’“”„•ˈ "
        words = [word.strip(excluded_symbols) for word in text.split()]
        words = list(OrderedDict.fromkeys(words).keys())

        with open('goemotions.tmp', 'w', encoding='utf-8') as f:
            for word in words:
                f.write(word + '\n')
            for s in excluded_symbols:
                f.write(s + '\n')

        tokenizer = utils.FullTokenizer('vocab.tmp')
        os.remove('vocab.tmp')
        return tokenizer

    @staticmethod
    def get_labels():
        pass

    @staticmethod
    def input_packets(batch):
        packet = utils.generate_bert_inputs(batch)
        if isinstance(batch, torch.Tensor):
            packet = {k: torch.tensor(v) for k, v in packet.items()}
        packet.update(input=batch)
        return packet
