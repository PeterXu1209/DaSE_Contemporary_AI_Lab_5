import torch.nn as nn
from transformers import AutoModel

class TextModel(nn.Module):

    def __init__(self, config):
        super(TextModel, self).__init__()

        self.bert = AutoModel.from_pretrained(config.bert_name)
        self.trans = nn.Sequential(
            nn.Dropout(config.bert_dropout),
            nn.Linear(self.bert.config.hidden_size, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )
        for param in self.bert.parameters():
            if config.fixed_text_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True
    def forward(self, bert_inputs, masks, token_type_ids=None):
        bert_out = self.bert(input_ids=bert_inputs, token_type_ids=token_type_ids, attention_mask=masks)
        hidden_state = bert_out['last_hidden_state']
        pooler_out = bert_out['pooler_output']

        return self.trans(hidden_state), self.trans(pooler_out)
