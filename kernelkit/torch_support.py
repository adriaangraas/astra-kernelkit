import torch
from torch.autograd import Function
import itertools
import cupy as cp

from astrapy.operator import Operator



class AutogradOperator(Function):
    """Autograd function for `Operator`."""

    @staticmethod
    def forward(ctx, input, op: Operator):
        B, C, _, _, _ = input.shape
        if input.requires_grad:
            ctx.operator = op
        output = input.new_empty(B, C, *op.range_shape, dtype=torch.float32)
        for s in itertools.product(range(B), range(C)):
            op(cp.asarray(input[s].detach()),
               out=cp.asarray(output[s].detach()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        op = ctx.operator
        B, C, _, _, _ = grad_output.shape
        grad_input = grad_output.new_zeros(B, C, *op.domain_shape,
                                           dtype=torch.float32)
        for s in itertools.product(range(B), range(C)):
            op.T(cp.asarray(grad_output[s].detach()),
                 out=cp.asarray(grad_input[s].detach()))
        return grad_input, None, None, None  # don't return gradient
