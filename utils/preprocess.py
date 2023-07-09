# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import numpy as np
from transformers import BertTokenizer
from .tokenization import FullTokenizer
from collections import OrderedDict


def pad_and_truncate(sequences, max_length, padding='post', truncating='post'):
    x = (np.zeros(max_length)).astype(dtype=np.int32)
    if truncating == 'pre':
        trunc = sequences[-max_length:]
    elif truncating == 'post':
        trunc = sequences[:max_length]
    else:
        raise NotImplementedError
    trunc = np.asarray(trunc, dtype=np.int32)
    if padding == 'pre':
        x[-len(trunc):] = trunc
    elif padding == 'post':
        x[:len(trunc)] = trunc
    else:
        raise NotImplementedError
    return x

def list_to_2darray(lst):
    max_length = max(len(sublst) for sublst in lst)
    result = np.zeros((len(lst), max_length))
    for i, row in enumerate(lst):
        result[i, :len(row)] = row
    return result

def onehot_encoding(array):
    types = np.unique(array)
    eye_matrix = np.eye(types.shape[0])
    return eye_matrix[array].sum(axis=1)[..., 1:]


class PretrainedBertTokenizer:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
