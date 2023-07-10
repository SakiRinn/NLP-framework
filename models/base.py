import torch.nn as nn
from abc import ABCMeta, abstractmethod


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, opt, loss):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.loss = loss

    def forward(self, input, gt=None):
        outputs = (None, ...)
        if gt is not None:
            loss = None
            outputs += (loss, )
        raise NotImplementedError
