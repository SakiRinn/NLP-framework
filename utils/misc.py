import os
import random
import logging
import sys
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


def set_logger(opt):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='[%(asctime)s | %(levelname)s | %(name)s]  %(message)s',
                        datefmt='%m.%d.%Y-%H:%M:%S',
                        level=logging.INFO)
    log_file = f'{opt.model}_{opt.dataset}_{strftime("%m%d%H%M", localtime())}.log'
    logger.addHandler(logging.FileHandler(log_file))

def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    results = dict()

    results["accuracy"] = accuracy_score(labels, preds)
    results["macro_precision"], results["macro_recall"], results[
        "macro_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="macro")
    results["micro_precision"], results["micro_recall"], results[
        "micro_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="micro")
    results["weighted_precision"], results["weighted_recall"], results[
        "weighted_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="weighted")

    return results
