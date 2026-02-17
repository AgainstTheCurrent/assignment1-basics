from collections.abc import Callable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                # Get iteration number from the state, or initial value.
                t = state.get("t", 0)
                # Get the gradient of loss with respect to p.
                grad = p.grad.data
                # Update weight tensor in-place.
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3,
                 weight_decay=0.01,
                 betas=(0.9, 0.999),
                 eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay,
                    "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                # Get iteration number from the state, or initial value.
                t = state.get("t", 1)
                # Get the gradient of loss with respect to p.
                grad = p.grad.data
                # Update the first moment estimate.
                m = state.get("m", torch.zeros_like(p.data))
                m = beta1 * m + (1 - beta1) * grad
                # Update the second moment estimate.
                v = state.get("v", torch.zeros_like(p.data))
                v = beta2 * v + (1 - beta2) * grad.pow(2)
                # Adjust learning rate for iteration t.
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                # Update weight tensor in-place.
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                # Apply weight decay.
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> Callable[[int], float]:
    """
    Returns a function that computes the learning rate at a given iteration
    using a cosine schedule with linear warmup.

    Args:
        it (int): The current iteration number.
        max_learning_rate (float): The maximum learning rate.
        min_learning_rate (float): The minimum learning rate.
        warmup_iters (int): The number of iterations to linearly increase the learning rate.
        cosine_cycle_iters (int): The number of iterations for one cosine cycle.

    Returns:
        Callable[[int], float]: A function that takes the current iteration and returns the learning rate.
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


""" Example usage:
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1e2)
for t in range(10):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.
print(weights)
"""
