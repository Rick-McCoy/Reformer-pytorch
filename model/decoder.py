import torch

from torch import nn
from model.feedforward import ChunkFeedForward
from model.attention import MultiRoundLSHAttention
from model.reversible import Reversible

class Decoder(nn.Module):
    def __init__(self, hp, args):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(hp, args) for _ in range(hp.model.N)])
    
    def forward(self, x1, x2, mask):
        for layer in self.layers:
            x1, x2 = Reversible.apply(layer, x1, x2, mask)
        return x2

class DecoderLayer(nn.Module):
    def __init__(self, hp, args):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiRoundLSHAttention(hp, args)
        self.feed_forward = ChunkFeedForward(hp, args)
        self.norm1 = nn.LayerNorm(hp.model.d_model)
        self.norm2 = nn.LayerNorm(hp.model.d_model)
        self.dropouts = nn.ModuleList([nn.Dropout(hp.model.dropout) for _ in range(2)])

    def forward(self, x1, x2, mask):
        y1 = x1 + self.dropouts[0](self.norm1(self.self_attn(x2, x2, x2, mask)))
        y2 = x2 + self.dropouts[1](self.norm2(self.feed_forward(y1)))
        return y1, y2

    def reverse(self, y1, y2, mask):
        x2 = y2 - self.dropouts[1](self.norm2(self.feed_forward(y1)))
        x1 = y1 - self.dropouts[0](self.norm1(self.self_attn(x2, x2, x2, mask)))
        return x1, x2
