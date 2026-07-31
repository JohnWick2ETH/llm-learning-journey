import os
import regex as re
from concurrent.futures import ProcessPoolExecutor
from pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def split_by_endoftext(f):
    """
    Split the input text by the "<|endoftext|>" and return an iterator of strings
    """
    until_eot = ""
    while True:
        text = f.read(4096)  # read 4KB at a time
        if text == "":
            # file is empty or we have reached the end of the file
            break
        if "<|endoftext|>" in text:
            parts = text.split("<|endoftext|>")
            until_eot += parts[0]

        ## if "<|endoftext|>" is in the text, we split it and yield the parts
        ## else if we only see a prefix of "<|endoftext|>", we need to look ahead to see
        # if the rest of the "<|endoftext|>" is in the next chunk

    return text.split("<|endoftext|>")


def pre_tokenize(text: bytes, special_tokens):
    """
    returns a map from bytestring to its frequency in the text.
    """
    freq_dict = {}
    text = text.decode("utf-8")  # convert bytes to Unicode string

    # 1. strip special_tokens out from text
    # 2. for each text part, run regex to extract pre-tokens
    strip_pat = "|".join([re.escape(token) for token in special_tokens])
    for sub_text in re.split(strip_pat, text):
        for s in re.finditer(PAT, sub_text):
            # convert Unicode string to utf-8 encoded bytes (pre-token)
            bs = s[0].encode("utf-8")
            freq_dict[bs] = freq_dict.get(bs, 0) + 1
    return freq_dict

def compute_merges(pre_tokens_dict, vocab_size):
    """
    Compute merges from pre tokens, by merge we mean pair of bytes.
    """
    init_vocab = set([b for b in range(256)])

    # TODO
    pass


def train_bpe(input_path, vocab_size, special_tokens):
    """
    Train a Byte Pair Encoding (BPE) tokenizer on the input text file.
    The output is (vocab, merges).
    """
    vocab = []

    with open(input_path, "rb") as f:
        # chunking the text by `special_token[0]` evenly among multiple threads
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=os.cpu_count(),
            split_special_token=special_tokens[0].encode("utf-8"),
        )
        # run pre-tokenization on each chunk in parallel
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # read chunk sequentially from file
            f.seek(start)
            chunk = f.read(end - start)
            chunks.append(chunk)
        with ProcessPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            st_chunks = special_tokens * len(chunks)
            freqs = list(executor.map(pre_tokenize, chunks, st_chunks))
            # merge N frequency dictionaries into one
            merged_freq = {}
            for freq in freqs:
                for pre_token, count in freq.items():
                    merged_freq[pre_token] = merged_freq.get(pre_token, 0) + count
            print(len(merged_freq))

    # compute merges

    # return vocuabulary and merges
    pass


def train_bpe_tinystories(data_folder):
    """
    Train a Byte Pair Encoding (BPE) tokenizer specifically for the TinyStories dataset.
    """
    train_bpe(
        input_path="%s/TinyStoriesV2-GPT4-valid.txt" % data_folder,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    pass


def train_bpe_expts_owt(data_folder):
    """
    Train a Byte Pair Encoding (BPE) tokenizer specifically for the OpenWebText dataset.
    """
    train_bpe(
        input_path="%s/owt_train.txt" % data_folder,
        vocab_size=50000,
        special_tokens=["<|endoftext|>"],
    )


if __name__ == "__main__":
    train_bpe(
        input_path="../data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
