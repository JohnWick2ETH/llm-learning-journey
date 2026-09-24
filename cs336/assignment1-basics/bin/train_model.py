import argparse
import json
import numpy as np
import torch
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.training import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.utils import try_gpu, cross_entropy_loss
from pathlib import Path


def main(args):
    # TODO: support incremental training
    # by loading previous checkpoint files
    # and then continue to finish

    token_path = args.tokens
    valid_token_path = args.valid_tokens
    vocab_size = args.vocab_size
    model_config_path = args.model_config
    training_config_path = args.training_config

    with open(model_config_path, "r") as f:
        mc = json.load(f)

    with open(training_config_path, "r") as f:
        tc = json.load(f)

    device = try_gpu()

    # init model's params
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=mc["context_length"],
        d_model=mc["d_model"],
        num_layers=mc["num_layers"],
        num_heads=mc["num_heads"],
        d_ff=mc["d_ff"],
        rope_theta=mc["rope_theta"],
        device=device,
    )

    # init optimizer
    optimizer = AdamW(
        params=model.parameters(),
        lr=tc["lr"],
        weight_decay=tc["weight_decay"],
        betas=tc["betas"],
        eps=tc["eps"],
    )

    checkpoint_path = Path(args.checkpoint_path)
    iter_count = 0
    if checkpoint_path.exists() and checkpoint_path.is_file():
        iter_count = load_checkpoint(args.checkpoint_path, model, optimizer)

    # load validation token set
    validation_set = np.memmap(valid_token_path, np.uint16)

    # load training tokens
    training_set = np.memmap(token_path, np.uint16)

    batch_size = tc["batch_size"]
    context_length = mc["context_length"]

    # TODO:
    # 1. checkpoint
    # 2. lazily load training dataset using mmap
    num_steps = len(training_set) // (batch_size * context_length)

    try:
        for step in range(iter_count, num_steps):
            optimizer.zero_grad(set_to_none=True)

            in_tokens, next_tokens = get_batch(
                training_set, batch_size, context_length, try_gpu()
            )

            # run forward pass to get the loss
            logits = model(in_tokens)
            loss = cross_entropy_loss(logits, next_tokens)

            # use the optimizer to adjust each param
            loss.backward()
            optimizer.step()

            iter_count = step + 1
            with torch.no_grad():
                # run the updated model on a small validation set
                valid_in, valid_next = get_batch(
                    validation_set,
                    batch_size,
                    context_length,
                    device,
                )
                valid_logits = model(valid_in)
                valid_loss = cross_entropy_loss(valid_logits, valid_next)
                print("loss on validation set at %dth step: %s" % (step, valid_loss))

    except KeyboardInterrupt:
        save_checkpoint(model, optimizer, iter_count, args.checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train_model")

    parser.add_argument("--tokens", required=True)
    parser.add_argument("--valid_tokens", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--vocab_size", required=True, type=int)
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--checkpoint_path", required=True)

    args = parser.parse_args()

    main(args)
