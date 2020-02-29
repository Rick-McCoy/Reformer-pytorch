import math
import torch

from torch import nn

class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, hp, args):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(hp.model.dropout)
        self.positional_encoding = self.get_encodings(hp.data.data_length, hp.model.d_model)

    def get_encodings(self, data_length, d_model) -> torch.Tensor:
        positional_encoding = torch.zeros(data_length, d_model)
        position = torch.arange(0, data_length, dtype=torch.float)[..., None]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)\
                             * -(math.log(10000) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding[None, ...]

    def forward(self, x):
        if self.positional_encoding.get_device() != x.get_device():
            self.positional_encoding = self.positional_encoding.to(device=x.get_device())
        x = x + self.positional_encoding[:, :x.size(1)].detach()
        return self.dropout(x)
