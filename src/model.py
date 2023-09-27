import torch.nn as nn 
import torch
from transformers import AutoModel

class Model(nn.Module):
    def __init__(self, bert, out_dim):
        super().__init__()
        self.bert = bert
        self.emb_dim = self.bert.embeddings.word_embeddings.weight.shape[-1]
        self.out = nn.Linear(self.emb_dim, out_dim)

    def forward(self, input):
        emb = self.bert(input).last_hidden_state[:, 0, :]
        emb = nn.functional.normalize(emb)
        return self.out(emb)


def load_model():
    model = torch.load('model/model.pth')
    return model