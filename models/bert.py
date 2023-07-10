from torch import nn
from .base import BaseModel
from transformers import BertConfig, BertModel


class BERT(BaseModel):
    def __init__(self, opt, bert=None, loss=nn.BCEWithLogitsLoss()):
        super(BERT, self).__init__(opt, loss)
        if bert is not None:
            self.bert = bert
        else:
            self.bert = BertModel(BertConfig())
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.hidden_size, opt.num_labels)

    def forward(self, input, gt=None, **kwargs):
        outputs = self.bert(input, **kwargs)
        pooled_output = self.dropout(outputs[1])
        logit = self.dense(pooled_output)

        # Add hidden states and attention if they are here
        outputs = (logit, ) + outputs[2:]

        if gt is not None:
            loss = self.loss(logit, gt)
            outputs += (loss, )
        return outputs
