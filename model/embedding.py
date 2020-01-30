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
        self.register_buffer(
            'pe', self.get_encodings(hp.data.data_length, hp.model.d_model)
        )

    def get_encodings(self, data_length, d_model) -> torch.Tensor:
        positional_encoding = torch.zeros(data_length, d_model)
        position = torch.arange(0, data_length, dtype=torch.float)[..., None]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)\
                             * -(math.log(10000) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding[None, ...]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)
