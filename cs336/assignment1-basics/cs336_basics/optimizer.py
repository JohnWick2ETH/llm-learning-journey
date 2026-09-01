from collections.abc import Callable
from typing import Optional
import torch
from math import sqrt, pow


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "lambda": weight_decay,
            "epsilon": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lbd = group["lambda"]
            eps = group["epsilon"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or 1.

                grad = p.grad.data  # Get the gradient of loss with respect to p.

                alpha = (
                    lr * sqrt(1 - pow(beta2, t)) / (1 - pow(beta1, t))
                )  # Compute adjusted 𝛼 for iteration t

                theta = (1 - lr * lbd) * p.data

                m = state.get("m", torch.zeros_like(p.data))
                m = beta1 * m + (1 - beta1) * grad

                v = state.get("v", torch.zeros_like(p.data))
                v = beta2 * v + (1 - beta2) * grad * grad

                p.data = theta - alpha * m / (torch.sqrt(v) + eps)

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1  # Increment iteration number.
        return loss
