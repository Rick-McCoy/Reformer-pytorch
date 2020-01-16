import torch
import torch.nn.functional as F

from torch import nn

class Feedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(Feedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        inter1 = F.relu(self.linear1(x))
        inter2 = self.linear2(self.dropout(x))
        return inter2

class ChunkFeedForward(nn.Module):
    def __init__(self, hp, args):
        super(ChunkFeedForward, self).__init__()
        self.chunk = hp.model.d_ff // hp.model.d_model
        self.linear1 = nn.Linear(hp.model.d_model, hp.model.d_ff)
        self.linear2 = nn.Linear(hp.model.d_ff, hp.model.d_model)
        self.dropout = nn.Dropout(hp.model.dropout)

    def forward(self, x):
        # [batch, length, d_model]
        x = x.reshape(-1, x.size(1) // self.chunk, x.size(2))
        # [batch * chunk, length // chunk, d_model]
        output = F.gelu(self.linear1(x))
        # [batch * chunk, length // chunk, d_ff]
        output = self.linear2(self.dropout(output))
        # [batch * chunk, length // chunk, d_model]
        output = output.reshape(-1, output.size(1) * self.chunk, output.size(2))
        # [batch, length, d_model]
        return output
