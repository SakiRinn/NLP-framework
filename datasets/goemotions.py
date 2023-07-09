import os
from typing import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset

import utils


class GoEmotions(Dataset):
    """Processor for the GoEmotions data set """

    def __init__(self, data_dir, label_file):
        self.data_dir = data_dir
        self.labels = self.get_labels(label_file)
        self.tokenizer = self.build_tokenizer()

    def __len__(self):
        return

    def get_labels(self, label_file):
        labels = []
        with open(os.path.join(self.args.data_dir, label_file), "r", encoding="utf-8") as f:
            for line in f:
                labels.append(line.rstrip())
        return labels

    def build_tokenizer(self):
        text = ''
        for file in self.data_dir:
            with open(file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
                for line in f:
                    text += line + ' '

        # words = [word.strip(r"#…!\"$%&'()*+,-—–./:;<=>?@[\\]^_`{|}~‘’“”„ˈ0123456789") for word in text.split()]
        words = text.split()
        words = list(OrderedDict.fromkeys(words).keys())

        with open('goemotions.tmp', 'w', encoding='utf-8') as f:
            for word in words:
                f.write(word + "\n")
            f.write('[NAME]\n')
            f.write('[RELIGION]\n')

        tokenizer = utils.FullTokenizer('goemotions.tmp', do_lower_case=False)
        os.remove('goemotions.tmp')
        return tokenizer

    def load_csv(self, filename):
        data = pd.read_csv(os.path.join(self.data_dir, filename), delimiter='\t').values
        sentences, labels = data[:, 0], data[:, 1]


        labels = np.char.split(labels.astype(str), sep=',')
        labels = utils.list_to_2darray(labels).astype(np.int32)
        one_hots = utils.onehot_encoding(labels)
        return sentences, one_hots