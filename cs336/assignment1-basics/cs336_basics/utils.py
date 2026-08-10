import torch


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
