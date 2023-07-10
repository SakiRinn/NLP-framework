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


def compute_metrics(pred, gt):
    assert len(pred) == len(gt)
    results = dict()

    results["accuracy"] = accuracy_score(gt, pred)
    results["macro_precision"], results["macro_recall"], results[
        "macro_f1"], _ = precision_recall_fscore_support(
        gt, pred, average="macro")
    results["micro_precision"], results["micro_recall"], results[
        "micro_f1"], _ = precision_recall_fscore_support(
        gt, pred, average="micro")
    results["weighted_precision"], results["weighted_recall"], results[
        "weighted_f1"], _ = precision_recall_fscore_support(
        gt, pred, average="weighted")

    return results
