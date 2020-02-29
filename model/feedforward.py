import torch
import torch.nn.functional as F

from torch import nn
from utils.utils import deterministic_dropout

class ChunkFeedForward(nn.Module):
    def __init__(self, hp, args):
        super(ChunkFeedForward, self).__init__()
        self.chunk = hp.model.d_ff // hp.model.d_model
        self.linear1 = nn.Linear(hp.model.d_model, hp.model.d_ff)
        self.linear2 = nn.Linear(hp.model.d_ff, hp.model.d_model)
        self.dropout = hp.model.dropout

    def forward(self, input_tensor, seed, random=True):
        # [batch, length, d_model]
        chunks = torch.chunk(input_tensor, chunks=self.chunk, dim=1)
        # [batch, length // chunk, d_model]
        output = [F.gelu(self.linear1(chunk)) for chunk in chunks]
        # [batch, length // chunk, d_ff]
        if self.training:
            output = [
                deterministic_dropout(chunk, seed + i, dropout=self.dropout)\
                    for chunk, i in zip(output, range(self.chunk))
            ]
            # [batch, length // chunk, d_ff]

        output = torch.cat([self.linear2(chunk) for chunk in output], dim=1)
        # [batch, length, d_model]
        return output
