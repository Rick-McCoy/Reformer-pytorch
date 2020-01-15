import torch

from torch import nn
from model.feedforward import ChunkFeedForward
from model.attention import MultiRoundLSHAttention
from model.reversible import Reversible

class Decoder(nn.Module):
    def __init__(self, hp, args):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([ReversibleDecoderLayer(hp, args) for _ in range(hp.model.N)])
    
    def forward(self, x1, x2, mask):
        for layer in self.layers:
            # x1, x2 = Reversible().apply(layer, x1, x2, mask)
            x1, x2 = layer(x1, x2, mask)
        return x2

class ReversibleDecoderLayer(nn.Module):
    def __init__(self, hp, args):
        super(ReversibleDecoderLayer, self).__init__()
        self.f_block = AttentionBlock(hp, args)
        self.g_block = FeedForwardBlock(hp, args)
    
    def forward(self, x1, x2, mask):
        y1 = x1 + self.f_block(x2, mask)
        y2 = x2 + self.g_block(y1)
        return y1, y2

class AttentionBlock(nn.Module):
    def __init__(self, hp, args):
        super(AttentionBlock, self).__init__()
        self.attn = MultiRoundLSHAttention(hp, args)
        self.norm = nn.LayerNorm(hp.model.d_model)
        self.dropout = nn.Dropout(hp.model.dropout)

    def forward(self, x, mask):
        norm = self.norm(x)
        return self.dropout(self.attn(norm, norm, mask))

class FeedForwardBlock(nn.Module):
    def __init__(self, hp, args):
        super(FeedForwardBlock, self).__init__()
        self.feed_forward = ChunkFeedForward(hp, args)
        self.norm = nn.LayerNorm(hp.model.d_model)
        self.dropout = nn.Dropout(hp.model.dropout)

    def forward(self, x):
        norm = self.norm(x)
        return self.dropout(self.feed_forward(norm))
