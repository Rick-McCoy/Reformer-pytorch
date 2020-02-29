import torch

from torch.autograd import Function

class Reversible(Function):

    def __init__(self):
        super(Reversible, self).__init__()

    @staticmethod
    def forward(ctx, *args):
        layer, input_1, input_2 = args
        ctx.layer = layer
        with torch.no_grad():
            output_1, output_2 = layer(input_1, input_2)
        Reversible.outputs = (output_1.detach(), output_2.detach())
        return output_1, output_2

    @staticmethod
    def backward(ctx, *grad_outputs):
        output_1_grad, output_2_grad = grad_outputs
        output_1, output_2 = Reversible.outputs
        output_1.requires_grad = True
        output_2.requires_grad = True

        with torch.enable_grad():
            g_output_1 = ctx.layer.g_block(output_1, ctx.layer.g_seed)
            g_output_1.backward(output_2_grad)

        with torch.no_grad():
            input_2 = output_2 - g_output_1
            del output_2, g_output_1
            input_1_grad = output_1_grad + output_1.grad
            del output_1_grad
            output_1.grad = None

        with torch.enable_grad():
            input_2.requires_grad = True
            f_input_2 = ctx.layer.f_block(input_2, ctx.layer.f_seed, False)
            f_input_2.backward(input_1_grad)

        with torch.no_grad():
            input_1 = output_1 - f_input_2
            del output_1, f_input_2
            input_2_grad = output_2_grad + input_2.grad
            del output_2_grad
            input_2.grad = None

            Reversible.outputs = (input_1.detach(), input_2.detach())

        return (None, input_1_grad, input_2_grad, None)
