import numpy as np
import torch
import os
import typing
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


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]
