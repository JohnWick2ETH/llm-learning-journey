import argparse
import logging
from dataclasses import dataclass
from cs336_basics.train_bpe import train_bpe, store_trained_artifacts
from time import perf_counter

logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    vocab_size: int
    special_tokens: list[str]


def main(
    input_file: str,
    data_set: str,
    vocab_size: int,
    special_tokens: list[str],
    artifacts_dir: str,
):
    start_t = perf_counter()
    vocab, merges = train_bpe(
        input_path=input_file,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    end_t = perf_counter()
    print("train %s took %.2fs" % (input_file, end_t - start_t))

    store_trained_artifacts(
        vocab,
        merges,
        "%s/%s_vocab.json" % (artifacts_dir, data_set),
        "%s/%s_merges.txt" % (artifacts_dir, data_set),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train_tokenizer")

    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
    )
    parser.add_argument("--data_set", required=True)
    parser.add_argument("--special_tokens", required=True)
    parser.add_argument("--artifacts", required=True)

    args = parser.parse_args()

    logging.basicConfig(
        filename="train_tokenizer_%s.log" % args.data_set,
        filemode="w",
        level=logging.INFO,
    )
    main(
        args.input, args.data_set, args.vocab_size, args.special_tokens, args.artifacts
    )
