import torch

from torch.autograd import Function

class Reversible(Function):

    def __init__(self):
        super(Reversible, self).__init__()

    @staticmethod
    def forward(ctx, *args):
        layer, x1, x2 = args
        ctx.layer = layer
        with torch.no_grad():
            y1, y2 = layer(x1, x2)
        Reversible.outputs = (y1.detach(), y2.detach())
        return y1, y2

    @staticmethod
    def backward(ctx, *grad_outputs):
        y1_grad, y2_grad = grad_outputs
        y1, y2 = Reversible.outputs
        y1.requires_grad = True
        y2.requires_grad = True

        with torch.enable_grad():
            gy1 = ctx.layer.g_block(y1, ctx.layer.g_seed)
            gy1.backward(y2_grad)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1
            x1_grad = y1_grad + y1.grad
            del y1_grad
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = ctx.layer.f_block(x2, ctx.layer.f_seed, False)
            fx2.backward(x1_grad)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2
            x2_grad = y2_grad + x2.grad
            del y2_grad
            x2.grad = None

            Reversible.outputs = (x1.detach(), x2.detach())

        return (None, x1_grad, x2_grad, None)
