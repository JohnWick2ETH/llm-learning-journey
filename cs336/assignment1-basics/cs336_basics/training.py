import numpy as np
import torch
from random import randint


def get_batch(
    dataset: np.typing.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    start_pos = [
        randint(0, len(dataset) - context_length - 1) for _ in range(batch_size)
    ]

    inputs = torch.from_numpy(
        np.array([dataset[sj : sj + context_length] for sj in start_pos])
    )
    inputs.to(device)
    nexts = torch.from_numpy(
        np.array([dataset[sj + 1 : sj + context_length + 1] for sj in start_pos])
    )
    nexts.to(device)
    return (inputs, nexts)
