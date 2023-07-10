import torch.nn as nn
import torch.optim as optim
import datasets
import models


def get_models():
    return {
        'bert': models.BERT,
        'lstm': models.LSTM,
    }


def get_datasets():
    return {
        'goemotions': datasets.GoEmotions
    }


def get_initializers():
    return {
        'xavier_uniform': nn.init.xavier_uniform_,
        'xavier_normal': nn.init.xavier_normal_,
        'orthogonal': nn.init.orthogonal_,
    }


def get_optimizers():
    return {
        'adadelta': optim.Adadelta,     # default lr=1.0
        'adagrad': optim.Adagrad,       # default lr=0.01
        'adam': optim.Adam,             # default lr=0.001
        'adamW': optim.AdamW,
        'adamax': optim.Adamax,         # default lr=0.002
        'asgd': optim.ASGD,             # default lr=0.01
        'rmsprop': optim.RMSprop,       # default lr=0.01
        'sgd': optim.SGD,
    }
