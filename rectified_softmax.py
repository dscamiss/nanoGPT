"""Implementation of rectified-softmax map."""

import torch

from profile_forward_backward import profile_forward_backward

# pylint: disable=W0221
# pylint: disable=W0223


class RectifiedSoftmax(torch.autograd.Function):
    """Implementation of rectified-softmax map."""

    thresh_hi = 0.9
    thresh_lo = 1.0 - thresh_hi
    slope = 0.05

    @staticmethod
    def forward(ctx, x):
        """
        computes rectified-softmax: TODO
        """

        thresh_hi = RectifiedSoftmax.thresh_hi
        thresh_lo = RectifiedSoftmax.thresh_lo

        means = torch.mean(x, dim=-1, keepdim=True)
        y = (1.0 / x.shape[-1]) * (1.0 + x - means)

        hi_mask = y > thresh_hi
        lo_mask = y < thresh_lo

        y[hi_mask] = thresh_hi + RectifiedSoftmax.slope * (y[hi_mask] - thresh_hi)
        y[lo_mask] = thresh_lo + RectifiedSoftmax.slope * (y[lo_mask] - thresh_lo)

        ctx.save_for_backward(hi_mask | lo_mask)

        return y

        # hi_idx = y > thresh_hi
        # lo_idx = y < thresh_lo
        # ctx.save_for_backward(lo_idx)

        #y_new = y.clone()
        # y_new[hi_idx] = thresh_hi + slope * (y_new[hi_idx] - thresh_hi)
        #print(f"lo_idx = {lo_idx}")
        #print(f"y_new[lo_idx] = {y_new[lo_idx]}")
        #y_new[lo_idx] = thresh_lo + slope * (y_new[lo_idx] - thresh_lo)
        #print(f"y (after mod) = {y_new}")

        #return y_new

    @staticmethod
    def backward(ctx, h):
        """
        Definition:
            rectified-softmax(x) = one_sc + (n_inv * x) - (n_inv * sum(x) * one_sc);

        Total derivative:
            d(rectified-softmax)(x).h = (1/n) h - (1/n) * sum(h) * 1_n

        Jacobian:
            J(rectified-softmax)(x) = diag(1_n) - 1_n 1_n^t

        Action on incoming gradients:
            e J(rectified-softmax)(x) = e diag(1_n) - e 1_n 1_n^t
                                      = e diag(1_n) - <e^t, 1_n> 1_n^t

        Threshold adjustment: TODO
        """

        grad_x = None
        if ctx.needs_input_grad[0]:
            hi_lo_idx = ctx.saved_tensors
            means = torch.mean(h, dim=-1, keepdim=True)
            grad_x = (1.0 / h.shape[-1]) * (h - means)
            grad_x[hi_lo_idx] = RectifiedSoftmax.slope * grad_x[hi_lo_idx]
        return grad_x

def main():
    """Test RectifiedSoftmax class."""
    rectified_softmax = RectifiedSoftmax.apply
    grad_check_flag = False
    profile_flag = True

    # Basic output check
    x = torch.tensor([
        [1, 2, 3],
        [5, 1, 3],
        [-8, -1, 0],
        [ 0.4823, 1.1401, -0.9511],
    ]).float()
    print(f"x = {x}")
    print(f"y = {rectified_softmax(x)}")

    # Gradient check
    if grad_check_flag:
        x = torch.randn((1, 3), dtype=torch.double, requires_grad=True)
        res = torch.autograd.gradcheck(rectified_softmax, x, eps=1e-6, atol=1e-6)  # type: ignore
        assert res, "Failed gradient check for rectified_softmax()"
        print("Passed gradient check for rectified_softmax()")

    # Code profile
    if profile_flag:
        x = torch.randn((10, 30), dtype=torch.double, requires_grad=True)
        profile_forward_backward(rectified_softmax, x)


if __name__ == "__main__":
    main()
