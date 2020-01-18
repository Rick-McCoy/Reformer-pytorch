import math
import torch

from torch import nn

class Embeddings(nn.Module):
    def __init__(self, hp, args):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(hp.data.vocab, hp.model.d_model)
        self.d_model = hp.model.d_model

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, hp, args):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(hp.model.dropout)

        pe = torch.zeros(hp.data.max_data_length, hp.model.d_model)
        position = torch.arange(0, hp.data.max_data_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hp.model.d_model, 2, dtype=torch.float)\
                 * -(math.log(10000) / hp.model.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)
