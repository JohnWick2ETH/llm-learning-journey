import heapq
import os
import regex as re
from concurrent.futures import ProcessPoolExecutor
from .pretokenization_example import find_chunk_boundaries
from dataclasses import dataclass

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@dataclass(slots=True)
class MaxHeapLexLargeEntry:
    frequency: int
    pair: tuple[bytes, bytes]

    def __lt__(self, other):
        return (self.frequency, self.pair) > (other.frequency, other.pair)


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


def compute_merges(
    pre_tokens_freq: dict[bytes, int], vocab_size, init_vocab_size
) -> list[tuple[bytes, bytes]]:
    """
    Compute merges from pre-tokens, by merge we mean pair of bytes.
    """
    num_vocabs = init_vocab_size
    merges = []

    # create a list to store pre_tokens s.t. we can access pre_token by index
    pre_tokens = list(pre_tokens_freq.keys())
    pre_token_to_idx = {pre_token: i for i, pre_token in enumerate(pre_tokens)}

    # 1. get the initial bytes pair frequency dict from pre_tokens_dict
    pair_freq = {}
    pair_to_pre_tokens = {}
    pre_token_decomp = {}  # each pre_token is decomposed as a sequence of symbols
    for pre_token, count in pre_tokens_freq.items():
        # pre_token is a bytes object
        pre_token_idx = pre_token_to_idx[pre_token]
        cur_decomp = [bytes([v]) for v in pre_token]
        pre_token_decomp[pre_token_idx] = cur_decomp
        for p0, p1 in zip(cur_decomp[:-1], cur_decomp[1:]):
            # pair_list stores all the pre-token that have it as substring
            # each item in the list has the form (pre_token_idx, pos, count)
            pair = (p0, p1)
            pair_list = pair_to_pre_tokens.setdefault(pair, [])
            # if a pair occurs multiple times in a pre-token
            if len(pair_list) > 0 and pair_list[-1][0] == pre_token_idx:
                old_count = pair_list[-1][1]
                pair_list[-1] = (pre_token_idx, old_count + count)
            else:
                pair_list.append((pre_token_idx, count))
            pair_freq[pair] = pair_freq.get(pair, 0) + count

    # convert pair frequency to max heap
    assert len(pair_freq) != 0
    pair_freq_heap = [
        MaxHeapLexLargeEntry(count, key) for key, count in pair_freq.items()
    ]
    heapq.heapify(pair_freq_heap)
    while num_vocabs < vocab_size:
        # pop out the most frequent pair
        most_frequent = heapq.heappop(pair_freq_heap)
        most_frequent_pair = most_frequent.pair
        merges.append(most_frequent_pair)
        num_vocabs += 1

        # [...,a,b,c,d,...] if [b,c] are merged, then we need to
        # 1. push [a,bc] to max-heap
        # 2. push [bc,d] to max-heap
        # 3. remove [b,c] from max-heap
        new_pairs_freq = {}
        pair_pre_tokens = pair_to_pre_tokens[most_frequent_pair]
        for pre_token_idx, count in pair_pre_tokens:
            pre_token = pre_tokens[pre_token_idx]
            cur_decomp = pre_token_decomp[pre_token_idx]
            # find a, d
            # find non-overlapping occurrences of [b,c] in pre-token-symbols
            pos = 0
            new_decomp = []
            old_pair = most_frequent_pair[0] + most_frequent_pair[1]
            while pos < len(cur_decomp) - 1:
                if (
                    cur_decomp[pos] == most_frequent_pair[0]
                    and cur_decomp[pos + 1] == most_frequent_pair[1]
                ):
                    # print(
                    #     "merge %s for pre_token %s with decomp %s"
                    #     % (old_pair, pre_token, cur_decomp)
                    # )
                    if pos > 0:
                        left = cur_decomp[pos - 1]
                        new_pair = (left, old_pair)
                        new_pairs_freq[new_pair] = (
                            new_pairs_freq.get(new_pair, 0) + count
                        )
                        new_pair_list = pair_to_pre_tokens.setdefault(new_pair, [])
                        new_pair_list.append((pre_token_idx, count))

                    if pos != len(cur_decomp) - 2:
                        right = cur_decomp[pos + 2]
                        new_pair = (old_pair, right)
                        new_pairs_freq[new_pair] = (
                            new_pairs_freq.get(new_pair, 0) + count
                        )
                        new_pair_list = pair_to_pre_tokens.setdefault(new_pair, [])
                        new_pair_list.append((pre_token_idx, count))

                    # replacing non-overlapping occurences of [b,c] by [bc]
                    new_decomp.append(old_pair)
                    pos += 2
                else:
                    new_decomp.append(cur_decomp[pos])
                    pos += 1  # check next symbol
            if pos == len(cur_decomp) - 1:
                new_decomp.append(cur_decomp[-1])
            # pre-token is replaced by new decomposition (with [b,c] replaced by [bc])
            pre_token_decomp[pre_token_idx] = new_decomp

        # insert [a,bc] and [bc,d] to pair frequency max heap
        for new_pair, count in new_pairs_freq.items():
            heapq.heappush(pair_freq_heap, MaxHeapLexLargeEntry(count, new_pair))

    return merges


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
):
    """
    Train a Byte Pair Encoding (BPE) tokenizer on the input text file.
    The output is (vocab, merges).
    """
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
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            st_chunks = special_tokens * len(chunks)
            freqs = list(executor.map(pre_tokenize, chunks, st_chunks))
            # merge N frequency dictionaries into one
            merged_freq = {}
            for freq in freqs:
                for pre_token, count in freq.items():
                    merged_freq[pre_token] = merged_freq.get(pre_token, 0) + count
        # compute merges
        merges = compute_merges(merged_freq, vocab_size, len(special_tokens) + 256)
        vocab_list = []
        vocab_list.extend(bytes([i]) for i in range(256))
        vocab_list.extend(st.encode() for st in special_tokens)
        vocab_list.extend((merge[0] + merge[1]) for merge in merges)

    vocab = {i: v for i, v in enumerate(vocab_list)}
    print(vocab.values())
    return vocab, merges


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
