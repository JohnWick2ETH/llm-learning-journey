import json
import regex as re
from collections.abc import Iterable
from cs336_basics.train_bpe import pre_tokenize, PAT


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and
            (optionally) a list of special tokens.
        """
        self.vocab = vocab  # token -> frequency
        self.merges = merges
        self.merge_priority = {
            pair: (len(merges) - i) for i, pair in enumerate(merges)
        }  # 0 is the lowest priority
        self.vocab_lkt = {token: id for id, token in vocab.items()}  # token id -> token
        self.special_tokens = sorted(special_tokens, key=len, reverse=True)

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens=None):
        """
        Construct a tokenizer from a given vocabulary file, merges file, and
            (optionally) a list of special tokens.
        """
        with open(vocab_path, encoding="utf-8") as f:
            serialized_vocab = json.load(f)

        vocab = {
            int(token_id): bytes.fromhex(token)
            for token, token_id in serialized_vocab.items()
        }

        with open(merges_path, encoding="utf-8") as f:
            merges = []
            for line in f:
                t = line.rstrip().split(" ")
                merges.append((bytes.fromhex(t[0]), bytes.fromhex(t[1])))

        return BPETokenizer(vocab, merges, special_tokens)

    def pre_tokenize(self, text: str) -> list[(bytes, bool)]:
        # find all occurrences of special_tokens
        if self.special_tokens is None:
            input_text = [text]
        else:
            find_pat = "|".join([re.escape(token) for token in self.special_tokens])
            input_text = re.split(r"(%s)" % find_pat, text)

        pre_tokens = []
        for sub_text in input_text:
            if self.special_tokens != None and sub_text in self.special_tokens:
                pre_tokens.append((sub_text.encode(), True))
            else:
                for s in re.finditer(PAT, sub_text):
                    # convert Unicode string to utf-8 encoded bytes (pre-token)
                    bs = s[0].encode("utf-8")
                    pre_tokens.append((bs, False))
        return pre_tokens

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of token IDs
        """
        # 0. if the input text has special tokens, we need to re
        # 1. pre-tokenize the input string into a list of pre-tokens (bytes)
        pre_tokens = self.pre_tokenize(text)
        token_ids = []
        # 2. for each pre-token, apply the merges in the order they were created
        for pre_token, is_special in pre_tokens:
            if is_special:
                print(pre_token)
                token_ids.append(self.vocab_lkt[pre_token])
                continue

            # 2.1 decompose the pre-token into a sequence of single-byte symbol.
            pre_token_decomp = [bytes([b]) for b in pre_token]

            while True:
                priority = 0 # 0 is the lowest priority
                # find pair that has highest merge priority
                for p0, p1 in zip(pre_token_decomp[:-1], pre_token_decomp[1:]):
                    cur = self.merge_priority.get((p0, p1))
                    if cur != None and cur > priority:
                        priority = cur
                        pair = (p0, p1)
                if priority == 0:
                    # no more applicable merge can be found
                    break

                # apply merge to derive new pre-token's decomposition
                new_decomp = []
                pos = 0
                while pos < len(pre_token_decomp) - 1:
                    if (
                        pre_token_decomp[pos + 0] == pair[0]
                        and pre_token_decomp[pos + 1] == pair[1]
                    ):
                        new_decomp.append(pair[0] + pair[1])
                        pos += 2
                    else:
                        new_decomp.append(pre_token_decomp[pos])
                        pos += 1
                if pos == len(pre_token_decomp) - 1:
                    new_decomp.append(pre_token_decomp[-1])

                # update pre_token_decomp
                pre_token_decomp = new_decomp

            # encode pre-token
            token_ids.extend([self.vocab_lkt[token] for token in pre_token_decomp])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Encode an iterable of strings into an iterable of token IDs
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a string
        """
        bs = b"".join([self.vocab[id] for id in ids])
        return bs.decode("utf-8", "replace")
