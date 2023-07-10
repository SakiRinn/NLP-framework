import glob
import os
from abc import ABCMeta, abstractmethod
from typing import OrderedDict

from torch.utils.data import Dataset

import utils


class BaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.tokenizer = self.build_tokenizer()
        self.labels = None

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    def build_tokenizer(self):
        text = ''
        for file in glob.glob(self.data_dir):
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
