import numpy as np

from torch import nn
from model.feedforward import ChunkFeedForward
from model.attention import MultiRoundLSHAttention
from model.reversible import Reversible
from utils.utils import deterministic_dropout

class Decoder(nn.Module):
    def __init__(self, hp, args):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([ReversibleDecoderLayer(hp, args) for _ in range(hp.model.N)])

    def forward(self, x1, x2):
        for layer in self.layers:
            layer.f_seed = int(np.random.randint(0, 1 << 63, dtype=np.int64))
            layer.g_seed = int(np.random.randint(0, 1 << 63, dtype=np.int64))
            x1, x2 = Reversible.apply(layer, x1, x2)
        return x2

class ReversibleDecoderLayer(nn.Module):
    def __init__(self, hp, args):
        super(ReversibleDecoderLayer, self).__init__()
        self.attn = MultiRoundLSHAttention(hp, args)
        self.feed_forward = ChunkFeedForward(hp, args)
        self.f_block = Block(hp, args, self.attn)
        self.g_block = Block(hp, args, self.feed_forward)

    def forward(self, x1, x2):
        y1 = x1 + self.f_block(x2, self.f_seed)
        y2 = x2 + self.g_block(y1, self.g_seed)
        return y1, y2

class Block(nn.Module):
    def __init__(self, hp, args, func):
        super(Block, self).__init__()
        self.func = func
        self.norm = nn.LayerNorm(hp.model.d_model)
        self.dropout = hp.model.dropout

    def forward(self, x, seed, random=True):
        norm = self.norm(x)
        out = self.func(norm, norm, (1 << 63) - seed, random)

        if self.training:
            return deterministic_dropout(out, seed=seed, dropout=self.dropout)

        return out
