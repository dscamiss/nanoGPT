"""Implementation of approximate-softmax map."""

import torch

from torch.autograd import Function


class ApproximateSoftmax(Function):
    @staticmethod
    def forward(ctx, x):
        # definition of approximate-softmax:
        # approximate-softmax(x) = 1_n + x/n - (sum(x)/n) 1_n,
        # where 1_n denotes the column vector whose entries are all 1/n.
        #
        # componentwise, we have:
        # approximate-softmax(x)^i = 1/n + (1/n) x^i - (1/n) sum(x) (1/n)
        #                          = (1/n) (1 + x^i - (1/n) sum(x)).

        n_inv = 1.0 / x.shape[-1]
        sums = n_inv * torch.sum(x, dim=-1, keepdim=True)
        softmax_approx_minus_n_inv = n_inv * (x - sums)
        softmax_approx = n_inv + softmax_approx_minus_n_inv
        # softmax_approx_norms = torch.linalg.vector_norm(softmax_approx, dim=-1, keepdim=True)
        softmax_approx_norms = softmax_approx.pow(2.0).sum(dim=-1, keepdim=True)
        y = softmax_approx_minus_n_inv / softmax_approx_norms
        return y
