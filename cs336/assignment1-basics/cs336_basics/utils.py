from collections.abc import Iterable
from math import cos, pi
import torch


def lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, Tw: int, Tc: int):
    if t < Tw:
        return t * alpha_max / Tw
    elif t <= Tc:
        return alpha_min + 0.5 * (1 + cos(pi * (t - Tw) / (Tc - Tw))) * (
            alpha_max - alpha_min
        )
    else:
        return alpha_min


def gradient_clipping_(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return

    with torch.no_grad():
        norms = torch.stack([g.norm(2) for g in grads])
        combined_norm = norms.norm(2)
        coeff = max_l2_norm / (combined_norm + 1e-6)
        coeff.clamp_(max=1.0)
        for g in grads:
            g.mul_(coeff)


# x[k] = e(x[k]) / \sum_k e(x[k])
def softmax(x: torch.Tensor, dim_i: int) -> torch.Tensor:
    x_max = torch.amax(x, dim_i, keepdim=True)
    # for numerical stability
    x_adjusted = x - x_max

    exp_x = torch.exp(x_adjusted)
    exp_x_sum = torch.sum(exp_x, dim=dim_i, keepdim=True)
    return exp_x / exp_x_sum


def silu(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x) * x


def log_softmax(x: torch.Tensor, dim_i: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    softmax(x) = e(x[i]-max) / sum_j e(x[j]-max)
    -log(softmax(x)) = log(sum_j e(x[j]-max)) - (x[i]-max)

    Args:
        x: The input tensor
        dim_i: The dimension along which to compute the log softmax.
    """
    x_max = torch.amax(x, dim_i, keepdim=True)
    # for numerical stability
    x_adjusted = x - x_max

    exp_x = torch.exp(x_adjusted)
    exp_x_sum = torch.sum(exp_x, dim=dim_i, keepdim=True)
    log_sum = torch.log(exp_x_sum)

    return (x_adjusted, log_sum)


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross-entropy loss between logits and targets.

    Args:
        logits: The predicted logits of shape (..., vocab_size)
        targets: The true next tokens of shape (..., )
    Returns:
        torch.Tensor: The average cross-entropy loss.
    """
    (x_adjusted, S) = log_softmax(logits, dim_i=-1)

    # x[i] - max(x) for i in targets
    target_x = x_adjusted.gather(dim=-1, index=targets.unsqueeze(-1))

    loss = (S - target_x).squeeze(-1)
    return loss.mean()
