import torch
from transformers import BertTokenizer, BertModel
from torch import nn

DEVICE = torch.device("cpu")

class FakeNewsModel(nn.Module):
    def __init__(self):
        super(FakeNewsModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.fc(output)

def load_model():
    model = FakeNewsModel()
    model.load_state_dict(torch.load("model/fake_news_model.pt", map_location=DEVICE))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer
