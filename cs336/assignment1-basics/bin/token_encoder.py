import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries


def encode_chunk(args) -> list[int]:
    tokenizer, file_path, start, end = args
    with open(file_path, "rb") as f:
        f.seek(start)
        ids = []
        while f.tell() < end:
            line = f.readline(end - start).decode("utf-8")
            ids.extend(tokenizer.encode(line))

    return ids


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

    num_tokens_per_line = 128
    with open(out_token_path, "w", encoding="utf-8") as f:
        for ids in token_ids:
            for s in range(0, len(ids), num_tokens_per_line):
                e = min(s + num_tokens_per_line, len(ids))
                # the output file is just lines of token ids
                f.write(" ".join(str(id) for id in ids[s:e]))
                f.write("\n")


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
