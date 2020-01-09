import torch

from torch.autograd import Function

class Reversible(Function):
    def __init__(self):
        super(Reversible, self).__init__()

    @staticmethod
    def forward(ctx, x1, x2, mask, f1, f2, x3=None, f3=None):
        if x3 is None:
            inter1 = x2 + f1(x2, x2, mask)
            inter2 = x1 + f2(x3, inter1, mask)
            inter3 = x2 + f3(inter2)
            return inter2, inter3
        inter1 = x1 + f1(x2)
        inter2 = x2 + f2(inter1)
        return inter1, inter2

    @staticmethod
    def backward(ctx, *grad_outputs):
        return super().backward(ctx, *grad_outputs)
