from argparse import ArgumentParser
import json
import timeit
import numpy as np
from cs336_basics.model import TransformerLM
from cs336_basics.data import get_batch
from cs336_basics.utils import try_gpu, cross_entropy_loss


def run(workload: str, model, optimizer, input, next):
    match workload:
        case "forward":
            model(input)
            pass
        case "forward_backward":
            logits = model(input)
            loss = cross_entropy_loss(logits=logits, targets=next)
            loss.backward()
            pass
        case "full":
            logits = model(input)
            loss = cross_entropy_loss(logits=logits, targets=next)
            loss.backward()
            optimizer.step()
            pass
        case _:
            raise ValueError(f"Unknown workload: {workload}")


def main(args):
    device = try_gpu()
    # 1. load hyperparameters configuration file
    assert args.model_size in [
        "small",
        "medium",
        "large",
        "xlarge",
        "10B",
    ], f"Unknown size: {args.model_size}"
    hyperparam = json.load(open(args.hyperparam, "r"))[args.model_size]

    # TODO: make context_length, vocab_size, batch_size configurable in hyperparam file
    context_length = 256
    vocab_size = 10000
    batch_size = 32

    # 2. load model and optimizer from hyperparameters
    model_param = hyperparam["model"]
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_param["d_model"],
        num_layers=model_param["num_layers"],
        num_heads=model_param["num_heads"],
        d_ff=model_param["d_ff"],
        rope_theta=model_param["rope_theta"],
        device=device,
    )

    # 3. generate random batch of data
    corpus = np.memmap(args.corpus_path, np.uint16)
    input, next = get_batch(
        corpus,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )

    # 4. run 3 benchmark cases
    workload = args.workload

    # w warmup steps first
    for _ in range(args.warmup_steps):
        run(workload, model, None, input, next)

    total_time = timeit.timeit(
        lambda: run(workload, model, None, input, next), number=10
    )
    print(
        "average %s pass for batch_size=%d, context_length=%d: %.4fs"
        % (workload, batch_size, context_length, total_time / 10)
    )
    pass

    pass


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmarking transformer model")

    parser.add_argument(
        "--hyperparam",
        type=str,
        required=True,
        help="hyperparam configuration file path",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        required=True,
        help="select param size from small, medium, large, xlarge, 10B",
    )
    parser.add_argument(
        "--corpus_path", type=str, required=True, help="path to the corpus token file"
    )
    parser.add_argument(
        "--workload", type=str, required=True, help="select workload to benchmark"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=10, help="number of warmup steps"
    )

    args = parser.parse_args()

    main(args)
