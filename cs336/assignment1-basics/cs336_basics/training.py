from dataclasses import dataclass

import json
import numpy as np
import torch
import os
import typing
from random import randint
from .model import TransformerLM
from .utils import cross_entropy_loss
from .optimizer import AdamW


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


@dataclass
class LMHyperParameter:

    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    batch_size: int
    lr: float
    weight_decay: float
    betas: tuple[float, float]

    @classmethod
    def from_json(jf: str):
        with open(jf, "r+") as file:
            param = json.load(file)

            return LMHyperParameter(
                vocab_size=param["vocab_size"],
                context_length=param["context_length"],
                d_model=param["d_model"],
                num_layers=param["num_layers"],
                num_heads=param["num_heads"],
                d_ff=param["d_ff"],
                rope_theta=param["rope_theta"],
                batch_size=param["batch_size"],
                lr=param["lr"],
            )


def train_lm_main(
    data_file: str, num_epochs: int, hyper_param: LMHyperParameter, checkpoint_file: str
):
    """
    main function to train transformer based language model

    Args:
        data_set: the path to the training data set
        checkpoint_file: the path to the checkpoint file
    """

    # load data set
    data_set = np.memmap(data_file, dtype=np.uint16)

    # init model hyperparameters like (lr)
    model = TransformerLM(
        vocab_size=hyper_param.vocab_size,
        context_length=hyper_param.context_length,
        d_model=hyper_param.d_model,
        num_layers=hyper_param.num_layers,
        num_heads=hyper_param.num_heads,
        d_ff=hyper_param.d_ff,
        rope_theta=hyper_param.rope_theta,
    )
    optimizer = AdamW(
        params=model.parameters(),
        lr=hyper_param.lr,
        weight_decay=hyper_param.weight_decay,
        betas=hyper_param.betas,
        eps=hyper_param.eps,
    )

    # training loop
    for _ in range(num_epochs):
        for _ in range(5):
            optimizer.zero_grad()
            inputs, nexts = get_batch(
                data_set,
                hyper_param.batch_size,
                context_length=hyper_param.context_length,
                device="cuda",
            )

            logits = model(inputs)
            loss = cross_entropy_loss(logits, nexts)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    train_lm_main()
