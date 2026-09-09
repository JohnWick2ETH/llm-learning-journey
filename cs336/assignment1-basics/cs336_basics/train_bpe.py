import heapq
import os
import json
import regex as re
import base64
import time
import logging
from concurrent.futures import ProcessPoolExecutor
from cs336_basics.pretokenization_example import find_chunk_boundaries
from dataclasses import dataclass
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MaxHeapLexLargeEntry:
    frequency: int
    pair: tuple[bytes, bytes]
    """
    turn a min-heap with lexicographical descending order
    into a max-heap with lexicographical ascending order
    """

    def __lt__(self, other):
        return (self.frequency, self.pair) > (other.frequency, other.pair)


def pre_tokenize(text: bytes, special_tokens, worker_id: int):
    """
    returns pre-tokens and their frequency in the text corpus
    """
    text = text.decode("utf-8")  # convert bytes to Unicode string
    pat_re = re.compile(PAT)

    # 1. strip `special_tokens` out from text
    # 2. for each text part, run regex to extract pre-tokens
    pre_tokens = {}
    strip_pat = "|".join([re.escape(token) for token in special_tokens])
    if len(special_tokens) == 0:
        stripped_text = [text]
    else:
        stripped_text = re.split(strip_pat, text)
    start_t = time.perf_counter()
    for sub_text in stripped_text:
        for s in pat_re.finditer(sub_text):
            bs = s[0]
            pre_tokens[bs] = pre_tokens.get(bs, 0) + 1
    end_t = time.perf_counter()
    if worker_id == 0:
        logger.info("pre_tokenize for worker 0 took %.2fs" % (end_t - start_t))

    # convert dict[string,int] to dict[bytes,int]
    pre_tokens = {
        pre_token.encode("utf-8"): freq for (pre_token, freq) in pre_tokens.items()
    }
    return pre_tokens


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

    pair_freq = {}  # always track the frequency of each pair
    pair_to_pre_tokens = {}  # track pre-tokens that have a given pair as substring
    pre_token_decomp = {}  # each pre_token is decomposed as a sequence of symbols

    # 1. get the initial pair frequency dict from pre_tokens_dict
    for pre_token, count in pre_tokens_freq.items():
        # pre_token is a bytes object
        pre_token_idx = pre_token_to_idx[pre_token]
        cur_decomp = [bytes([v]) for v in pre_token]
        pre_token_decomp[pre_token_idx] = cur_decomp

        pairs = Counter()
        for p0, p1 in zip(cur_decomp[:-1], cur_decomp[1:]):
            pair = (p0, p1)
            pairs[pair] += count

        for pair, count in pairs.items():
            # pair_list stores all the pre-token that have it as substring
            # each item in the list has the form (pre_token_idx, count)
            pair_freq[pair] = pair_freq.get(pair, 0) + count
            pair_to_pre_tokens.setdefault(pair, []).append(pre_token_idx)

    # 2. convert pair frequency to max heap
    assert len(pair_freq) != 0
    pair_freq_heap = [
        MaxHeapLexLargeEntry(count, key) for key, count in pair_freq.items()
    ]
    heapq.heapify(pair_freq_heap)

    # 3. iteratively pop out the most frequent pair as new merge
    #    and update the pair frequency dict and max heap
    while num_vocabs < vocab_size:
        start_t = time.perf_counter()
        # pop out the most frequent pair as new merge
        # it's possible that a pair has multiple frequency records
        # in the max-heap, so called "stale" records,
        # we need to pop out stale records until we find the most recent one
        # the accurate frequency of a pair is stored in `pair_freq` dict
        while True:
            most_frequent = heapq.heappop(pair_freq_heap)
            if pair_freq[most_frequent.pair] == most_frequent.frequency:
                break
        most_frequent_pair = most_frequent.pair
        merges.append(most_frequent_pair)
        num_vocabs += 1

        # track the frequency change of each pair after applying this merge
        pair_freq_diffs = {}

        # compute new decomposition: replace non-overlapping occurences of [b,c] by [bc]
        num_related_pre_tokens = len(pair_to_pre_tokens[most_frequent_pair])
        # TODO: parallelize the below work
        for pre_token_idx in pair_to_pre_tokens[most_frequent_pair]:
            pre_token = pre_tokens[pre_token_idx]
            pre_token_count = pre_tokens_freq[pre_token]

            # decompositions of pre-token before/after applying merge
            cur_decomp = pre_token_decomp[pre_token_idx]
            new_decomp = []

            merge = most_frequent_pair[0] + most_frequent_pair[1]
            pos = 0
            while pos < len(cur_decomp) - 1:
                if (
                    cur_decomp[pos] == most_frequent_pair[0]
                    and cur_decomp[pos + 1] == most_frequent_pair[1]
                ):
                    new_decomp.append(merge)
                    pos += 2
                else:
                    new_decomp.append(cur_decomp[pos])
                    pos += 1
            if pos == len(cur_decomp) - 1:
                new_decomp.append(cur_decomp[-1])

            # counter for old pairs and new pairs in the pre-token
            old_pairs = Counter()
            new_pairs = Counter()
            for p0, p1 in zip(cur_decomp[:-1], cur_decomp[1:]):
                old_pairs[(p0, p1)] += pre_token_count
            for p0, p1 in zip(new_decomp[:-1], new_decomp[1:]):
                new_pairs[(p0, p1)] += pre_token_count

            # update pre-token's decomposition
            pre_token_decomp[pre_token_idx] = new_decomp

            # update pair_to_pre_tokens
            for pair, _ in old_pairs.items():
                if pair not in new_pairs:
                    # old pair no longer exists in this pre-token
                    # so we need to remove it from the list
                    pair_to_pre_tokens[pair] = [
                        idx for idx in pair_to_pre_tokens[pair] if idx != pre_token_idx
                    ]
            for pair, _ in new_pairs.items():
                if pair not in old_pairs:
                    # new pair occurs in the new decomposition but not in the old one
                    # we need to add it to pair_to_pre_tokens
                    pair_to_pre_tokens.setdefault(pair, []).append(pre_token_idx)

            new_pairs.subtract(old_pairs)
            for pair, delta_count in new_pairs.items():
                pair_freq_diffs[pair] = pair_freq_diffs.get(pair, 0) + delta_count

        # update pair_freq and pair_freq_heap
        for pair, freq_count_diff in pair_freq_diffs.items():
            new_freq = pair_freq.get(pair, 0) + freq_count_diff
            pair_freq[pair] = new_freq
            heapq.heappush(pair_freq_heap, MaxHeapLexLargeEntry(new_freq, pair))
        end_t = time.perf_counter()
        logger.info(
            "get vocab %d which occurs in %d pre-tokens took %.2fs"
            % (num_vocabs, num_related_pre_tokens, end_t - start_t)
        )

    return merges


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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
        start_t = time.perf_counter()
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            st_chunks = [special_tokens] * len(chunks)
            worker_ids = [i for i in range(len(chunks))]
            freqs = list(executor.map(pre_tokenize, chunks, st_chunks, worker_ids))
            # merge N frequency dictionaries into one
            merged_freq = {}
            for freq in freqs:
                for pre_token, count in freq.items():
                    merged_freq[pre_token] = merged_freq.get(pre_token, 0) + count
        end_t = time.perf_counter()
        logger.info(
            "pre_tokenize over %d workers in %.2fs" % (len(chunks), end_t - start_t)
        )
        # compute merges
        merges = compute_merges(merged_freq, vocab_size, len(special_tokens) + 256)
        vocab_list = []
        vocab_list.extend(bytes([i]) for i in range(256))
        vocab_list.extend(st.encode() for st in special_tokens)
        vocab_list.extend((merge[0] + merge[1]) for merge in merges)

    vocab = {i: v for i, v in enumerate(vocab_list)}
    return vocab, merges


def store_trained_artifacts(vocab, merges, vocab_path, merges_path):
    """
    Store the trained BPE artifacts (vocab and merges) to disk.
    """
    with open(vocab_path, "w", encoding="utf-8") as f:
        # store each token as base64-encoded string to avoid issues with non-UTF-8 bytes
        json.dump(
            {
                base64.b64encode(token).decode("utf-8"): token_id
                for token_id, token in vocab.items()
            },
            f,
            ensure_ascii=False,
        )
    with open(merges_path, "w", encoding="utf-8") as f:
        for token1, token2 in merges:
            f.write(
                "%s %s\n"
                % (
                    base64.b64encode(token1).decode("utf-8"),
                    base64.b64encode(token2).decode("utf-8"),
                )
            )
