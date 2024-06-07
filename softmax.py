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
        return y

