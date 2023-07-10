import os
import random
import logging
from time import strftime, localtime

import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_metrics(pred, gt, zero_division=1):
    assert len(pred) == len(gt)
    results = dict()

    results["accuracy"] = accuracy_score(gt, pred)
    results["macro_precision"], results["macro_recall"], results[
        "macro_f1"], _ = precision_recall_fscore_support(
        gt, pred, average="macro", zero_division=zero_division)
    results["micro_precision"], results["micro_recall"], results[
        "micro_f1"], _ = precision_recall_fscore_support(
        gt, pred, average="micro", zero_division=zero_division)
    results["weighted_precision"], results["weighted_recall"], results[
        "weighted_f1"], _ = precision_recall_fscore_support(
        gt, pred, average="weighted", zero_division=zero_division)

    return results


def array_to_text(tokenizer, indices):
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy()
    pad_indices = np.where(indices == 0)[0]
    if len(np.where(indices == 0)[0]) != 0:
        indices = indices[:pad_indices[0]]
    tokens = tokenizer.convert_ids_to_tokens(indices)
    return tokenizer.convert_tokens_to_string(tokens)
