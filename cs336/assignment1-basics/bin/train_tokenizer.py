import argparse
import json
from dataclasses import dataclass
from cs336_basics.train_bpe import train_bpe


@dataclass
class TokenizerConfig:
    vocab_size: int
    special_tokens: list[str]


def main(input_file: str, vocab_size: int, special_tokens: list[str], output_file: str):
    vocab, merges = train_bpe(
        input_path=input_file,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    # TODO: encode vocab as dict[bytes, int] and store it to vocab.json

    # TODO: store merges.txt


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train_tokenizer")

    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--vocab_size",
        required=True,
    )
    parser.add_argument("--special_tokens", required=True)
    parser.add_argument("--output")

    args = parser.parse_args()
    main(args.input, args.vocab_size, args.special_tokens, args.output)
