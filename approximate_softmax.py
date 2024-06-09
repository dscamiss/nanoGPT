"""Implementation of approximate-softmax map."""

import torch

# pylint: disable=W0221
# pylint: disable=W0223


class ApproximateSoftmax(torch.autograd.Function):
    """Implementation of approximate-softmax map with explicit backward()."""

    @staticmethod
    def first_order_softmax_approx(x, subtract_n_inv: bool = False):
        """
        computes the first-order taylor series approximation to softmax at 0:
            t(x) = 1_n + x/n - (sum(x)/n) 1_n = 1_n + x/n - mean(x) 1_n,
        where 1_n is the column vector whose entries are all 1/n.
        """

        means = torch.mean(x, dim=-1, keepdim=True)
        if torch.any(means.isnan()):
            print(x)
        if subtract_n_inv:
            y = (x - means) / x.shape[-1]
        else:
            y = (1.0 + x - means) / x.shape[-1]
        return y

    @staticmethod
    def forward(ctx, x):
        """
        computes approximate-softmax:
            approximate-softmax(x) = (t(x) - 1_n) / <t(x), t(x)>,
        where 1_n is the column vector whose entries are all 1/n and
        t(x) is the first-order taylor series approximation to softmax at 0.
        """

        n_inv = 1.0 / x.shape[-1]
        t_x = ApproximateSoftmax.first_order_softmax_approx(x)
        t_x_norm_sq = t_x.pow(2.0).sum(dim=-1, keepdim=True)
        y = (t_x - n_inv) / t_x_norm_sq
        ctx.save_for_backward(t_x, t_x_norm_sq, y)
        return y

    @staticmethod
    def backward(ctx, h):
        """
        total derivative:
            d(approximate-softmax)(x).h = TODO
        """

        grad_x = None
        if ctx.needs_input_grad[0]:
            t_x, t_x_norm_sq, y = ctx.saved_tensors
            t_h = ApproximateSoftmax.first_order_softmax_approx(h, True)
            z = t_h / t_x_norm_sq
            inner_product = torch.sum(z * t_x, dim=-1, keepdim=True)
            grad_x = z - 2.0 * (inner_product * y)
        return grad_x


def main():
    """Test ApproximateSoftmax class."""
    approximate_softmax = ApproximateSoftmax.apply

    # Basic output check
    x = torch.tensor([[1, 2, 3], [5, 1, 3], [-8, -1, 0]]).float()
    y = approximate_softmax(x)
    print(f"x = {x}")
    print(f"y = {y}")

    # Gradient check
    x = torch.randn((10, 30), dtype=torch.double, requires_grad=True)
    res = torch.autograd.gradcheck(approximate_softmax, x, eps=1e-6, atol=1e-6)  # type: ignore
    assert res, "Failed gradient check for approximate_softmax()"
    print("Passed gradient check for approximate_softmax()")


if __name__ == "__main__":
    main()
