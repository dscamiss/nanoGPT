"""Implementation of rectified-softmax map."""

import torch
from torch import nn

from profile_forward_backward import profile_forward_backward

# pylint: disable=W0221
# pylint: disable=W0223

# TODO: Timing experiments, masking adjustments


class RectifiedSoftmax(nn.Module):
    """Implementation of rectified-softmax map."""

    def __init__(self, rectify_high=False, negative_slope=0.01):
        super().__init__()
        self.rectify_high = rectify_high
        self.negative_slope = negative_slope
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.first_order_softmax_approx = FirstOrderSoftmaxApprox.apply

    def forward(self, x):
        """Forward pass."""
        y = self.first_order_softmax_approx(x)
        y = self.leaky_relu(y)
        if self.rectify_high:
            y = y - (1.0 - self.negative_slope) * self.relu(y - 1.0)
        return y


class FirstOrderSoftmaxApprox(torch.autograd.Function):
    """Implementation of first-order softmax approximation map."""

    @staticmethod
    def forward(ctx, x):
        means = torch.mean(x, dim=-1, keepdim=True)
        return 1.0 + x - means

    @staticmethod
    def backward(ctx, h):
        grad_x = None
        if ctx.needs_input_grad[0]:
            means = torch.mean(h, dim=-1, keepdim=True)
            grad_x = h - means
        return grad_x


def main():
    """Test FirstOrderSoftmaxApproximation class."""

    first_order_softmax_approx = FirstOrderSoftmaxApprox.apply
    grad_check_flag = True
    profile_flag = False

    # Basic output check
    x = torch.tensor(
        [
            [1, 2, 3],
            [5, 1, 3],
            [-8, -1, 0],
            [0.4823, 1.1401, -0.9511],
        ]
    ).float()
    print(f"x = {x}")
    print(f"y = {first_order_softmax_approx(x)}")

    # Gradient check
    if grad_check_flag:
        x = torch.randn((10, 30), dtype=torch.double, requires_grad=True)
        res = torch.autograd.gradcheck(
            first_order_softmax_approx, x, eps=1e-6, atol=1e-6
        )
        assert res, "Failed gradient check for first_order_softmax_approx()"
        print("Passed gradient check for first_order_softmax_approx()")

    # Code profile
    if profile_flag:
        x = torch.randn((10, 30), dtype=torch.double, requires_grad=True)
        profile_forward_backward(first_order_softmax_approx, x)


if __name__ == "__main__":
    main()
