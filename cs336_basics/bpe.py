from collections import Counter, defaultdict
from itertools import pairwise
import logging
from multiprocessing import Pool
import os
from typing import Iterable

import regex as re
from sortedcontainers import SortedDict, SortedSet


EOS = "<|endoftext|>"
PRE_TOKENIZATION_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

logger = logging.getLogger(__name__)


class MergeCache:
    def __init__(self):
        self.token_pair_counts = Counter()
        self.count_to_pairs = SortedDict()
        self.token_to_pre_tokens = defaultdict(set)
        self.pre_tokens = Counter()

    def add_pair(self, tok1: bytes, tok2: bytes, count: int):
        pair = (tok1, tok2)
        self.token_pair_counts[pair] += count
        new_count = self.token_pair_counts[pair]
        if new_count not in self.count_to_pairs:
            s = SortedSet()
            s.add(pair)
            self.count_to_pairs[new_count] = s
        else:
            self.count_to_pairs[new_count].add(pair)
        old_count = new_count - count
        if old_count > 0:
            pairs = self.count_to_pairs[old_count]
            pairs.remove(pair)
            if not pairs:
                del self.count_to_pairs[old_count]

    def add_pre_token(self, pre_token: tuple[bytes, ...]):
        self.pre_tokens[pre_token] += 1

    def get_pair_to_merge(self) -> tuple[tuple[bytes, bytes], int] | None:
        if not self.count_to_pairs:
            return None
        count, pairs = self.count_to_pairs.peekitem()
        logger.info(f"Pairs: {pairs}")
        pair = pairs.pop()
        pairs.add(pair)
        return pair, count

    def merge(self, tok1: bytes, tok2: bytes, count: int):
        pair = (tok1, tok2)
        new_token = tok1 + tok2
        pre_tokens_to_visit = self.token_to_pre_tokens[tok1] & self.token_to_pre_tokens[tok2]
        for pre_token in pre_tokens_to_visit:
            pre_token_count = self.pre_tokens[pre_token]
            assert pre_token_count > 0
            did_merge = False
            merged_indexes = set()
            for pre_token_idx, (pre_token_tok_1, pre_token_tok_2) in enumerate(pairwise(pre_token)):
                if did_merge:
                    merged_indexes.add(pre_token_idx)
                    did_merge = False
                    continue
                if pre_token_tok_1 != tok1 or pre_token_tok_2 != tok2:
                    did_merge = False
                    continue
                if pre_token_idx > 0:
                    if pre_token_idx - 1 in merged_indexes:
                        tok_to_update = new_token
                    else:
                        tok_to_update = pre_token[pre_token_idx - 1]
                    self.remove_pair(tok_to_update, tok1, pre_token_count)
                    self.add_pair(tok_to_update, new_token, pre_token_count)
                if pre_token_idx + 2 < len(pre_token):
                    self.remove_pair(tok2, pre_token[pre_token_idx + 2], pre_token_count)
                    self.add_pair(new_token, pre_token[pre_token_idx + 2], pre_token_count)
                did_merge = True
            if did_merge:
                merged_indexes.add(pre_token_idx + 1)
            if not merged_indexes:
                continue
            new_pre_token = []
            for tok_idx, tok in enumerate(pre_token):
                self.token_to_pre_tokens[tok].discard(pre_token)
                if tok_idx in merged_indexes:
                    new_pre_token.pop()
                    new_pre_token.append(new_token)
                else:
                    new_pre_token.append(tok)
            new_pre_token = tuple(new_pre_token)
            for tok in new_pre_token:
                self.token_to_pre_tokens[tok].add(new_pre_token)
            self.pre_tokens[new_pre_token] = pre_token_count
            del self.pre_tokens[pre_token]
        remaining = self.token_pair_counts.pop(pair)
        pairs_for_count = self.count_to_pairs[remaining]
        pairs_for_count.remove(pair)
        if not pairs_for_count:
            del self.count_to_pairs[remaining]

    def remove_pair(self, tok1: bytes, tok2: bytes, pre_token_count: int):
        pair = (tok1, tok2)
        count = self.token_pair_counts[pair]
        pairs_for_count = self.count_to_pairs[count]
        pairs_for_count.remove(pair)
        if not pairs_for_count:
            del self.count_to_pairs[count]
        new_count = count - pre_token_count
        if new_count > 0:
            self.token_pair_counts[pair] = new_count
            if new_count not in self.count_to_pairs:
                s = SortedSet()
                s.add(pair)
                self.count_to_pairs[new_count] = s
            else:
                self.count_to_pairs[new_count].add(pair)
        else:
            del self.token_pair_counts[pair]


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    with open(input_path, "rb") as f:
        content = f.read().decode("utf-8", errors="ignore")
    logger.info("Content read")
    merge_cache = MergeCache()

    for pre_token in _pre_tokenize(content, special_tokens):
        merge_cache.add_pre_token(pre_token)
    logger.info("Pre-tokenization finished")
    for pre_token, pre_token_count in merge_cache.pre_tokens.items():
        for tok1, tok2 in pairwise(pre_token):
            merge_cache.add_pair(tok1, tok2, pre_token_count)
            merge_cache.token_to_pre_tokens[tok1].add(pre_token)
        if len(pre_token) > 1:
            merge_cache.token_to_pre_tokens[tok2].add(pre_token)
    logger.info("Merge cache initialized")

    merges: list[tuple[bytes, bytes]] = []
    vocab = _init_vocab(special_tokens)
    while len(vocab) < vocab_size:
        pair_to_merge = merge_cache.get_pair_to_merge()
        logger.info("Pair to merge: %s", pair_to_merge)
        if pair_to_merge is None:
            break
        pair, count = pair_to_merge
        tok1, tok2 = pair
        merge_cache.merge(tok1, tok2, count)
        new_tok = tok1 + tok2
        vocab[len(vocab)] = new_tok
        merges.append(pair)
    return vocab, merges


def _pre_tokenize(text: str, special_tokens: list[str]) -> Iterable[tuple[bytes, ...]]:
    if len(special_tokens) == 0:
        for pre_token_match in re.finditer(PRE_TOKENIZATION_REGEX, text):
            pre_token = pre_token_match.group()
            yield tuple(bytes([ch]) for ch in pre_token.encode())
    escaped_specials = sorted((re.escape(t) for t in special_tokens), key=len, reverse=True)
    split_pattern = "|".join(escaped_specials)
    documents = re.split(split_pattern, text)
    for doc in documents:
        if doc in special_tokens:
            yield tuple(bytes([ch]) for ch in doc.encode())
        for pre_token_match in re.finditer(PRE_TOKENIZATION_REGEX, doc):
            pre_token = pre_token_match.group()
            yield tuple(bytes([ch]) for ch in pre_token.encode())


def _init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {
        i: bytes([i]) for i in range(256)
    }
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    return vocab
