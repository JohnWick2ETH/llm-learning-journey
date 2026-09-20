import argparse
import json
import numpy as np
import torch
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.training import get_batch
from cs336_basics.utils import cross_entropy_loss


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def load_token(token_path: str) -> np.typing.NDArray:
    tokens = []
    with open(token_path, "r", encoding="utf-8") as f:
        while True:
            inp = f.readline()
            if inp == "":
                break
            tokens.extend(int(id) for id in inp.split(" ") if id != "")
    return np.array(tokens)


def main(args):
    token_path = args.tokens
    valid_token_path = args.valid_tokens
    vocab_size = args.vocab_size
    model_config_path = args.model_config
    training_config_path = args.training_config

    with open(model_config_path, "r") as f:
        mc = json.load(f)

    with open(training_config_path, "r") as f:
        tc = json.load(f)

    # init model's params
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=mc["context_length"],
        d_model=mc["d_model"],
        num_layers=mc["num_layers"],
        num_heads=mc["num_heads"],
        d_ff=mc["d_ff"],
        rope_theta=mc["rope_theta"],
        device=try_gpu(),
    )

    # init optimizer
    optimizer = AdamW(
        params=model.parameters(),
        lr=tc["lr"],
        weight_decay=tc["weight_decay"],
        betas=tc["betas"],
        eps=tc["eps"],
    )

    # load validation token set
    validation_set = load_token(valid_token_path)
    print("validation set loaded")

    # load training tokens
    training_set = load_token(token_path)
    print("training set loaded")

    # train model in a few epochs
    num_epochs = tc["num_epochs"]
    batch_size = tc["batch_size"]
    context_length = mc["context_length"]
    for e in range(num_epochs):
        # in each epoch, keep sampling small batches
        num_samples = len(training_set) // (batch_size * context_length)
        for i in range(num_samples):
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

            with torch.no_grad():
                # run the updated model on a small validation set
                valid_in, valid_next = get_batch(
                    validation_set,
                    batch_size,
                    context_length,
                    try_gpu(),
                )
                valid_logits = model(valid_in)
                valid_loss = cross_entropy_loss(valid_logits, valid_next)
                print("loss on validation set at sample %d: %s" % (i, valid_loss))

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train_model")

    parser.add_argument("--tokens", required=True)
    parser.add_argument("--valid_tokens", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--vocab_size", required=True, type=int)
    parser.add_argument("--training_config", required=True)

    args = parser.parse_args()

    main(args)
