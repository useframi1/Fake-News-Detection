import torch.nn as nn
import torch
from transformers import BertModel


class BertBiLSTMClassifier(nn.Module):
    def __init__(self, config: dict, num_classes: int):
        super(BertBiLSTMClassifier, self).__init__()

        self.config = config

        self.bert = BertModel.from_pretrained(
            self.config["bert_model"], hidden_dropout_prob=self.config["bert_dropout"]
        )
        self.bert.gradient_checkpointing_enable()
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * self.config["hidden_size"], num_classes)
        self.dropout = nn.Dropout(self.config["lstm_out_dropout"])

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = bert_output.last_hidden_state

        lstm_out, _ = self.lstm(last_hidden_state)
        lstm_out = torch.mean(lstm_out, dim=1)

        output = self.fc(self.dropout(lstm_out))
        return output
