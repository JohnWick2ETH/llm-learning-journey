from argparse import ArgumentParser
import json
import timeit
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch


def main(args):
    # 1. load hyperparameters configuration file
    assert args.size in [
        "small",
        "medium",
        "large",
        "xlarge",
        "10B",
    ], f"Unknown size: {args.size}"
    hyperparam = json.load(open(args.hyperparam, "r"))[args.size]

    # TODO: make context_length and vocab_size configurable in hyperparam file
    context_length = 1024
    vocab_size = 10000

    # 2. load model and optimizer from hyperparameters
    model_param = hyperparam["model"]
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_param["d_model"],
        num_layers=model_param["num_layers"],
        num_heads=model_param["num_heads"],
        d_ff=model_param["d_ff"],
    )

    # 3. generate random batch of data
    batch_size = args.batch_size
    # batch = get_batch(, batch_size=batch_size, context_length=context_length, device="cuda")

    # 4. run 3 benchmark cases
    workload = args.workload
    match workload:
        case "forward":
            # run forward pass only

            # w warmup steps first
            for _ in range(args.warmup_steps):
                model(batch)

            timeit.timeit(lambda: model(batch), number=10)
            pass
        case "forward_backward":
            # run forward + backward pass
            pass
        case "full":
            # run forward + backward + optimizer step
            pass
        case _:
            raise ValueError(f"Unknown workload: {workload}")

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
        "--size",
        type=str,
        required=True,
        help="select param size from small, medium, large, xlarge, 10B",
    )
    parser.add_argument(
        "--workload", type=str, required=True, help="select workload to benchmark"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=10, help="number of warmup steps"
    )

    args = parser.parse_args()

    main(args)
