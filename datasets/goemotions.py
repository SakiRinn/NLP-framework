import glob
import os
from typing import OrderedDict

import numpy as np
import pandas as pd

import utils
from .base import BaseDataset


class GoEmotions(BaseDataset):

    def __init__(self, data_dir, label_file='labels.txt', mode='all', max_seq_len=50):
        super(GoEmotions, self).__init__(data_dir)
        self.labels = self.get_labels(label_file)
        self.max_seq_len = max_seq_len

        if mode == 'all':
            train_data, train_gt = self.load_tsv('train.tsv')
            test_data, test_gt = self.load_tsv('test.tsv')
            dev_data, dev_gt = self.load_tsv('dev.tsv')
            self.data = np.concatenate([train_data, test_data, dev_data], axis=0)
            self.gt = np.concatenate([train_gt, test_gt, dev_gt], axis=0)
        elif mode == 'val':
            self.data, self.gt = self.load_tsv('dev.tsv')
        else:
            self.data, self.gt = self.load_tsv(mode + '.tsv')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.gt[index]

    def build_tokenizer(self):
        text = ''
        for tsv in glob.glob(os.path.join(self.data_dir, '*.tsv')):
            sentences = pd.read_csv(tsv, delimiter='\t').values[:, 0]
            for sentence in sentences:
                text += sentence + ' '
        text = text.lower()
        text = text.replace('[name]', '[NAME]').replace('[religion]', '[RELIGION]')

        excluded_symbols = "#…!\"$%&'()*+,-—–./:;<=>?@[\\]^_`{|}~‘’“”„•ˈ "
        words = [word.strip(excluded_symbols) for word in text.split()]
        words = list(OrderedDict.fromkeys(words).keys())

        with open('vocab.tmp', 'w', encoding='utf-8') as f:
            for word in words:
                f.write(word + '\n')
            for s in excluded_symbols:
                f.write(s + '\n')
            f.write('[NAME]\n')
            f.write('[RELIGION]\n')

        tokenizer = utils.FullTokenizer('vocab.tmp', do_lower_case=True)
        os.remove('vocab.tmp')
        return tokenizer

    def get_labels(self, label_file):
        labels = []
        with open(os.path.join(self.data_dir, label_file), "r", encoding="utf-8") as f:
            for line in f:
                labels.append(line.strip())
        return labels

    def load_tsv(self, filename):
        data = pd.read_csv(os.path.join(self.data_dir, filename), delimiter='\t').values
        sentences, labels = data[:, 0], data[:, 1]

        tokens = [self.tokenizer.tokenize(sentence) for sentence in sentences]
        indices = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]
        indices = utils.indices_to_array(indices, self.max_seq_len)

        labels = np.char.split(labels.astype(str), sep=',')
        labels = utils.indices_to_array(labels)
        one_hots = utils.onehot_encoding(labels)
        return indices, one_hots
