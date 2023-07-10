import os
import numpy as np


def indices_to_array(labels, max_seq_len=0):
    if max_seq_len <= 0:
        max_seq_len = max(len(lst) for lst in labels)
    result = np.zeros((len(labels), max_seq_len), dtype=np.int32)
    for i, lst in enumerate(labels):
        length = min(len(lst), max_seq_len)
        result[i, :length] = np.int32(lst[:length]) + 1
    return result


def onehot_encoding(array):
    types = np.unique(array)
    eye_matrix = np.eye(types.shape[0])
    return eye_matrix[array].sum(axis=1)[..., 1:].astype(np.float32)


def generate_bert_inputs(array):
    return {
        'attention_mask': np.where(array != 0, 1, 0),
        'token_type_ids': np.zeros_like(array),
        'position_ids': np.broadcast_to(np.arange(array.shape[1]), array.shape)
    }
