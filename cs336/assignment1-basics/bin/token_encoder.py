import argparse


def main(
    vocab_file_path: str,
    merges_file_path: str,
    special_tokens: list[str],
    input_file_path: str,
):
    with open(vocab_file_path, "r") as vf:
        pass
    pass


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
    parser.add_argument("--output")
    parser.add_argument("--dtype", choices=["uint16", "uint32"])

    args = parser.parse_args()

    # main()
