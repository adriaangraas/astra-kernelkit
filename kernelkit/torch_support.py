from itertools import product

import cupy as cp
import torch
from torch.autograd import Function

from kernelkit.operator import BaseOperator, XrayTransform


class AutogradOperator(Function):
    """Autograd function for `Operator` objects."""

    @staticmethod
    def forward(ctx, input, op: BaseOperator):
        B, C, _, _, _ = input.shape
        if input.requires_grad:
            ctx.operator = op
        out = input.new_empty(B, C, *op.range_shape, dtype=torch.float32)
        for s in product(range(B), range(C)):
            op(cp.asarray(input[s].detach()), out=cp.asarray(out[s].detach()))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        op = ctx.operator
        B, C, _, _, _ = grad_output.shape
        # TODO(Adriaan): allocation here may not be CUDA Graph safe
        grad_input = grad_output.new_zeros(B, C, *op.domain_shape, dtype=torch.float32)
        for s in product(range(B), range(C)):
            op.T(
                cp.asarray(grad_output[s].detach()),
                out=cp.asarray(grad_input[s].detach()),
            )
        return grad_input, None, None, None  # don't return gradient


class XrayForwardProjection(torch.nn.Module):
    """PyTorch module for `XrayTransform`."""

    def __init__(self, *args, **kwargs):
        super(XrayForwardProjection, self).__init__()
        self.op = XrayTransform(*args, **kwargs)

    def forward(self, input):
        return AutogradOperator.apply(input, self.op)


class XrayBackprojection(torch.nn.Module):
    """PyTorch module for `XrayTransform`."""

    def __init__(self, *args, **kwargs):
        super(XrayBackprojection, self).__init__()
        self.op = XrayTransform(*args, **kwargs)

    def forward(self, input):
        return AutogradOperator.apply(input, self.op.T)
