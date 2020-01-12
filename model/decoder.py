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
            x1, x2 = layer(x1, x2, mask)
        return x2

class DecoderLayer(nn.Module):
    def __init__(self, hp, args):
        super(DecoderLayer, self).__init__()
        self.attn = ReversibleAttention(hp, args)
        self.feed_forward = ReversibleFeedforward(hp, args)

    def forward(self, x1, x2, mask):
        y1, x2 = Reversible().apply(self.attn, x1, x2, mask)
        y1, y2 = Reversible().apply(self.feed_forward, y1, x2, mask)
        return y1, y2

class ReversibleAttention(nn.Module):
    def __init__(self, hp, args):
        super(ReversibleAttention, self).__init__()
        self.self_attn = MultiRoundLSHAttention(hp, args)
        self.norm = nn.LayerNorm(hp.model.d_model)
        self.dropout = nn.Dropout(hp.model.dropout)
    
    def forward(self, x1, x2, mask):
        y1 = x1 + self.dropout(self.norm(self.self_attn(x2, x2, mask)))
        return y1, x2

    def reverse(self, y1, x2, mask):
        x1 = y1 - self.dropout(self.norm(self.self_attn(x2, x2, mask)))
        return x1, x2

class ReversibleFeedforward(nn.Module):
    def __init__(self, hp, args):
        super(ReversibleFeedforward, self).__init__()
        self.feed_forward = ChunkFeedForward(hp, args)
        self.norm = nn.LayerNorm(hp.model.d_model)
        self.dropout = nn.Dropout(hp.model.dropout)

    def forward(self, y1, x2, mask):
        y2 = x2 + self.dropout(self.norm(self.feed_forward(y1)))
        return y1, y2

    def reverse(self, y1, y2, mask):
        x2 = y2 - self.dropout(self.norm(self.feed_forward(y1)))
        return y1, x2
