import argparse
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries


def encode_chunk(args) -> np.typing.NDArray[np.uint16]:
    tokenizer, file_path, start, end = args
    with open(file_path, "rb") as f:
        f.seek(start)
        ids = []
        while f.tell() < end:
            line = f.readline(end - start).decode("utf-8")
            ids.extend(tokenizer.encode(line))

    return np.array(ids, dtype=np.uint16)


def main(
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str] | None,
    corpus_path: str,
    out_token_path: str,
):
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)

    num_processes = os.cpu_count()
    with open(corpus_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, num_processes, b"\n")

    chunks = [
        (tokenizer, corpus_path, start, end)
        for (start, end) in zip(chunk_boundaries[:-1], chunk_boundaries[1:])
    ]
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # TODO: handle the case corpus is very large (e.g. 500+ GB)
        # it will easily exceed the main memory limit
        token_ids = executor.map(encode_chunk, chunks)

    with open(out_token_path, "wb") as f:
        for ids in token_ids:
            f.write(ids.tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="token_encoder",
    )

    parser.add_argument(
        "--vocab",
        required=True,
    )
    parser.add_argument(
        "--merges",
        required=True,
    )
    parser.add_argument(
        "--input",
        required=True,
    )
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    main(
        args.vocab,
        args.merges,
        "<|endoftext|>",
        args.input,
        args.output,
    )
