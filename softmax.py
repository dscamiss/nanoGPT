"""Implementation of softmax map."""

import torch

from torch.autograd import Function


class Softmax(Function):
    @staticmethod
    def forward(ctx, x):
        # definition of softmax:
        # softmax(x)^i = exp(x^i) / sum(exp(x^i)).
        #
        # translation-invariance property:
        # softmax(x)^i = exp(x^i - c) / sum(exp(x^i - c)).

        maxes = torch.max(x, dim=-1, keepdim=True)[0]
        x_exp = torch.exp(x - maxes)
        x_exp_sum = torch.sum(x_exp, dim=-1, keepdim=True)
        y = x_exp / x_exp_sum
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_in):
        # total derivative:
        # d(softmax)(x).h = h - <softmax(x), h> 1.
        #
        # jacobian matrix:
        # J(softmax)(x) = diag(softmax(x)) - softmax(x) softmax(x)^t,
        # where "diag(x)" is the diagonalization of x and "x^t" is the transpose of x.
        #
        # action on incoming gradients:
        # e J(softmax)(x) = e diag(softmax(x)) - e softmax(x) softmax(x)^t
        #                 = e * softmax(x) - <e^t, softmax(x)> softmax(x)^t,
        # where "x * y" is the elementwise product of x and y, and <x, y> is the
        # euclidean inner product of x and y.
        #
        # note: the derivations above assume that softmax(x) is a column vector.
        # in the implementation below, softmax(x) = y is a row vector,
        # so there is no need for the transpose operation(s) when computing the
        # action on incoming gradients.

        grad_x = None
        if ctx.needs_input_grad[0]:
            y, = ctx.saved_tensors
            elementwise_product = grad_in * y
            inner_product = torch.sum(elementwise_product, dim=-1, keepdim=True)
            grad_x = elementwise_product - inner_product * y
        return grad_x
