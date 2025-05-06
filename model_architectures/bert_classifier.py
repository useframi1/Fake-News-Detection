import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, config: dict, num_classes: int):
        super(BertClassifier, self).__init__()

        self.config = config

        self.bert = BertModel.from_pretrained(self.config["bert_model"])
        self.bert.gradient_checkpointing_enable()
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(self.config["dropout"])

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output

        output = self.fc(self.dropout(pooled_output))
        return output
