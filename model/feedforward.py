import torch
import torch.nn.functional as F

from torch import nn

class ChunkFeedForward(nn.Module):
    def __init__(self, hp, args):
        super(ChunkFeedForward, self).__init__()
        self.chunk = hp.model.d_ff // hp.model.d_model
        self.linear1 = nn.Linear(hp.model.d_model, hp.model.d_ff)
        self.linear2 = nn.Linear(hp.model.d_ff, hp.model.d_model)
        self.dropout = hp.model.dropout

    def forward(self, x, seed):
        # [batch, length, d_model]
        x = x.reshape(-1, x.size(1) // self.chunk, x.size(2))
        # [batch * chunk, length // chunk, d_model]
        output = F.gelu(self.linear1(x))
        # [batch * chunk, length // chunk, d_ff]
        if self.training:
            generator = torch.Generator(device=output.get_device())
            generator.manual_seed(seed)
            dropout_mask = torch.bernoulli(output, p=1 - self.dropout, generator=generator)
            output = dropout_mask * output / (1 - self.dropout)

        output = self.linear2(output)
        # [batch * chunk, length // chunk, d_model]
        output = output.reshape(-1, output.size(1) * self.chunk, output.size(2))
        # [batch, length, d_model]
        return output
