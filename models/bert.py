from torch import nn
from .base import BaseModel
from transformers import BertConfig, BertModel


class BERT(BaseModel):
    def __init__(self, opt, loss=nn.BCEWithLogitsLoss(), bert=None):
        super(BERT, self).__init__(opt, loss)
        if bert is not None:
            self.bert = bert
        else:
            self.bert = BertModel(BertConfig())
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.hidden_size, opt.num_labels)

    def forward(self, input, gt=None, *,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None):
        outputs = self.bert(
            input,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = self.dropout(outputs[1])
        logit = self.dense(pooled_output)

        # Add hidden states and attention if they are here
        outputs = (logit, ) + outputs[2:]

        if gt is not None:
            loss = self.loss(logit, gt)
            outputs += (loss, )

        return outputs
