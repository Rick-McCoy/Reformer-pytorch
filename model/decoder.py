import torch
import numpy as np

from torch import nn
from model.feedforward import ChunkFeedForward
from model.attention import MultiRoundLSHAttention
from model.reversible import Reversible

class Decoder(nn.Module):
    def __init__(self, hp, args):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([ReversibleDecoderLayer(hp, args) for _ in range(hp.model.N)])

    def forward(self, x1, x2, mask):
        Reversible.mask = mask
        for layer in self.layers:
            layer.f_seed = np.random.randint(0, 1 << 63, dtype=np.int64).item()
            layer.g_seed = np.random.randint(0, 1 << 63, dtype=np.int64).item()
            x1, x2 = Reversible.apply(layer, x1, x2)
        return x2

class ReversibleDecoderLayer(nn.Module):
    def __init__(self, hp, args):
        super(ReversibleDecoderLayer, self).__init__()
        self.f_block = AttentionBlock(hp, args)
        self.g_block = FeedForwardBlock(hp, args)

    def forward(self, x1, x2, mask):
        y1 = x1 + self.f_block(x2, mask, self.f_seed)
        y2 = x2 + self.g_block(y1, self.g_seed)
        return y1, y2

class AttentionBlock(nn.Module):
    def __init__(self, hp, args):
        super(AttentionBlock, self).__init__()
        self.attn = MultiRoundLSHAttention(hp, args)
        self.norm = nn.LayerNorm(hp.model.d_model)
        self.dropout = hp.model.dropout

    def forward(self, x, mask, seed, random=True):
        norm = self.norm(x)
        attn = self.attn(norm, norm, mask, random)

        if self.training:
            generator = torch.Generator(device=attn.get_device())
            generator.manual_seed(seed)
            mask = torch.bernoulli(attn, p=self.dropout, generator=generator)
            return mask * attn / (1 - self.dropout)

        return attn

class FeedForwardBlock(nn.Module):
    def __init__(self, hp, args):
        super(FeedForwardBlock, self).__init__()
        self.feed_forward = ChunkFeedForward(hp, args)
        self.norm = nn.LayerNorm(hp.model.d_model)
        self.dropout = hp.model.dropout

    def forward(self, x, seed):
        norm = self.norm(x)
        feed_forward = self.feed_forward(norm)

        if self.training:
            generator = torch.Generator(device=feed_forward.get_device())
            generator.manual_seed(seed)
            mask = torch.bernoulli(feed_forward, p=self.dropout, generator=generator)
            return mask * feed_forward / (1 - self.dropout)

        return feed_forward
