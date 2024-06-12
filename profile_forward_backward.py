"""Profile forward and backward passes."""

import time
import torch


def profile_forward_backward(module, x):
    """Profile forward and backward passes."""
    # Warmup
    for _ in range(50):
        output = module(x)

    g0 = torch.rand_like(output)
    for _ in range(50):
        output = module(x)
        output.backward(g0)

    nb_iters = 100

    # Profile forward pass
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ]
    ) as p:
        for _ in range(nb_iters):
            output = module(x)

    print("--- forward profile ---")
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))

    # Profile backward pass
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ]
    ) as p:
        start = time.time()
        for _ in range(nb_iters):
            output = module(x)
            # no weights
            # module.weight.grad = None
            output.backward(g0)
        end = time.time()
        fwd_time = (end - start) / nb_iters

    print("--- backward profile ---")
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
    print(f"time per iter: {fwd_time}")
