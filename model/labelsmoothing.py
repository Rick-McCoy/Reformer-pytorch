import torch
import torch.nn.functional as F

from torch import nn

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        dist = torch.empty_like(pred).fill_(self.smoothing / (pred.size(-1) - 1))
        dist.scatter_(dim=-1, index=target.unsqueeze(-1), value=(1 - self.smoothing))
        loss = F.kl_div(log_prob, dist)
        return loss
