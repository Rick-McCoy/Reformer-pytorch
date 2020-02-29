import torch
import torch.nn.functional as F

from torch import nn

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing, vocab, chunk):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.chunk = chunk
        self.vocab = vocab

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        pred = pred.flatten(0, 1)
        target = target.flatten(0, 1)
        mask = mask.flatten(0, 1).float()
        chunked_pred = torch.chunk(pred, chunks=self.chunk, dim=0)
        chunked_target = torch.chunk(target, chunks=self.chunk, dim=0)
        chunked_mask = torch.chunk(mask, chunks=self.chunk, dim=0)
        log_prob = [F.log_softmax(p, dim=-1) for p in chunked_pred]
        loss = [self.smoothed_loss(p, t, m)[None]\
            for p, t, m in zip(log_prob, chunked_target, chunked_mask)]
        loss = torch.cat(loss, dim=0).sum()
        return loss / mask.sum()

    def smoothed_loss(self, log_prob: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        dist = torch.full_like(log_prob, fill_value=self.smoothing / (self.vocab - 2))
        dist.scatter_(dim=-1, index=target[..., None], value=(1 - self.smoothing))
        dist[..., -1] = 0
        dist *= mask[..., None]
        log_prob = log_prob * mask[..., None]
        loss = F.kl_div(log_prob, dist, reduction='sum')
        return loss
