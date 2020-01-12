import torch

from torch.autograd import Function

class Reversible(Function):
    
    outputs = None

    def __init__(self):
        super(Reversible, self).__init__()

    @staticmethod
    def forward(ctx, *args):
        function, x1, x2, mask = args
        ctx.function = function
        ctx.mask = mask
        with torch.no_grad():
            y1, y2 = ctx.function(x1, x2, mask)
        Reversible.outputs = (y1, y2)
        return y1, y2

    @staticmethod
    def backward(ctx, *grad_outputs):
        y1_grad, y2_grad = grad_outputs
        y1, y2 = Reversible.outputs
        mask = ctx.mask
        x1, x2 = ctx.function.reverse(y1, y2, mask)
        Reversible.outputs = (x1, x2)
        with torch.enable_grad():
            if not x1.requires_grad:
                x1.requires_grad = True
            if not x2.requires_grad:
                x2.requires_grad = True
            y1, y2 = ctx.function(x1, x2, mask)
        grad = torch.autograd.grad(outputs=(y1, y2), inputs=(x1, x2), grad_outputs=grad_outputs)
        return (None, *grad, None)
