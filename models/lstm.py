import torch
import torch.nn as nn

from .base import BaseModel
from .layers.dynamic_rnn import DynamicLSTM


class LSTM(BaseModel):
    def __init__(self, opt, embedding=None, loss=nn.BCEWithLogitsLoss()):
        super(LSTM, self).__init__(opt, loss)
        if embedding is not None:
            self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding, dtype=torch.float))
        else:
            self.embed = nn.Embedding(opt.vocab_len, opt.embed_size)
        self.lstm = DynamicLSTM(opt.embed_size, opt.hidden_size, num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt.hidden_size, opt.num_labels)

    def forward(self, input, gt=None):
        x = self.embed(input)
        x_len = torch.sum(input != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        logit = self.dense(h_n[0])
        outputs = (logit, h_n)

        if gt is not None:
            loss = self.loss(logit, gt)
            outputs += (loss, )
        return outputs
