import torch
import numpy as np
from typing import List, Union, Dict, Tuple, Optional
from collections import defaultdict
from time import time
import heapq
import json
import re
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import setup_logger

logger = setup_logger(__name__)


class BPETokenizer:
    def __init__(self, vocab_file: Optional[str] = None):
        """Byte Pair Encoding (BPE) Tokenizer — flat character-level, no pre-splitting.

        Args:
            vocab_file: Optional path to a JSON file previously saved with
                        ``save()``.  When provided the tokenizer is fully
                        restored (vocab + merges) and is ready to encode/decode
                        without calling ``fit()`` first.
        """
        self.str2int: Dict[str, int] = {}
        self.int2str: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.eos_token = None
        self.unk_token = None

        if vocab_file is not None:
            self.load(vocab_file)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the full tokenizer state (vocab + merges) to a JSON file.

        The file can be passed to ``__init__`` or ``load()`` to restore the
        tokenizer exactly, including the ability to encode new text.

        Args:
            path: Destination file path (e.g. ``"tokenizer.json"``).
        """
        state = {
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "vocab": self.str2int,
            "merges": self.merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved tokenizer to '{path}' ({len(self.str2int)} tokens, {len(self.merges)} merges)")

    def load(self, path: str) -> None:
        """Restore tokenizer state from a JSON file produced by ``save()``.

        Calling this (or passing ``vocab_file`` to ``__init__``) overwrites any
        existing state.  A subsequent call to ``fit()`` will also overwrite it.

        Args:
            path: Path to the JSON file.
        """
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.eos_token = state["eos_token"]
        self.unk_token = state["unk_token"]
        self.str2int = {tok: int(idx) for tok, idx in state["vocab"].items()}
        self.int2str = {int(idx): tok for tok, idx in state["vocab"].items()}
        self.merges = [tuple(pair) for pair in state["merges"]]
        logger.info(f"Loaded tokenizer from '{path}' ({len(self.str2int)} tokens, {len(self.merges)} merges)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_on_special_tokens(self, text: str) -> List[Tuple[str, bool]]:
        """Partition *text* into (chunk, is_special) pairs.

        Special tokens are matched literally and returned as atomic chunks;
        everything else is returned as regular text to be BPE-encoded.
        """
        special_tokens = [t for t in [self.eos_token, self.unk_token] if t is not None]
        if not special_tokens:
            return [(text, False)]

        # Sort by length descending to prevent partial matches
        special_tokens.sort(key=len, reverse=True)

        pattern = "(" + "|".join(re.escape(s) for s in special_tokens) + ")"
        special_set = set(special_tokens)
        
        return [(p, p in special_set) for p in re.split(pattern, text) if p]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        text_data: Union[str, List[str]],
        vocab_size: int,
        min_freq: Optional[int] = None,
        eos_token: str = "<|EOS|>",
        unk_token: str = "<|UNK|>",
        verbose: bool = True,
    ):
        """Fit the tokenizer using flat BPE (no word pre-splitting).

        The entire text is treated as one continuous character sequence.
        Spaces and newlines are regular tokens and will participate in merges
        exactly like any other character, so the full text structure is
        preserved through encode → decode round-trips.

        Args:
            text_data:        Training text (str) or list of texts joined by
                              endoftext_token.
            vocab_size:       Target vocabulary size; training stops when reached.
            min_freq:         Stop early when the most-frequent remaining pair
                              appears fewer than this many times.
            eos_token:         Added to vocab; used as document separator when
                              text_data is a list.
            unk_token:        Added to vocab; used for chars unseen at train time.
            verbose:          Show a tqdm progress bar during training.
        """
        if isinstance(text_data, list):
            text = eos_token.join(text_data)
        else:
            text = text_data

        # ---- Initialise vocabulary ----------------------------------------
        self.eos_token = eos_token
        self.unk_token = unk_token
        base_chars = sorted(set(text))
        vocab: List[str] = [eos_token, unk_token] + base_chars

        self.str2int = {tok: i for i, tok in enumerate(vocab)}
        self.int2str = {i: tok for tok, i in self.str2int.items()}
        self.merges = []

        # ---- Build a doubly-linked list over the character sequence --------
        # prev_arr[i] / next_arr[i] = neighbouring *valid* indices
        # tokens[i] = None means position i has been merged away
        n = len(text)
        tokens: List[Optional[str]] = list(text)
        prev_arr: List[int] = list(range(-1, n - 1))   # prev_arr[0] = -1
        next_arr: List[int] = list(range(1, n + 1))    # next_arr[n-1] = n

        # ---- Count all adjacent pairs + remember their positions -----------
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_positions: Dict[Tuple[str, str], set] = defaultdict(set)

        for i in range(n - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += 1
            pair_positions[pair].add(i)

        # ---- Max-heap via negated counts (lazy deletion for stale entries) -
        heap: List[Tuple[int, Tuple[str, str]]] = [
            (-cnt, pair) for pair, cnt in pair_counts.items()
        ]
        heapq.heapify(heap)

        # ---- BPE merge loop ------------------------------------------------
        t0 = time()
        target_merges = vocab_size - len(vocab)

        with tqdm(
            total=target_merges,
            desc="BPE merges",
            unit="merge",
            disable=not verbose,
        ) as pbar:
            while len(self.str2int) < vocab_size:
                # Pop until we find a heap entry whose count is still current
                best_pair: Optional[Tuple[str, str]] = None
                best_count = 0
                while heap:
                    neg_cnt, candidate = heapq.heappop(heap)
                    actual = pair_counts.get(candidate, 0)
                    if actual == -neg_cnt and actual > 0:
                        best_pair = candidate
                        best_count = actual
                        break

                if best_pair is None:
                    break
                if min_freq is not None and best_count < min_freq:
                    break

                new_token = best_pair[0] + best_pair[1]
                if new_token not in self.str2int:
                    idx = len(self.str2int)
                    self.str2int[new_token] = idx
                    self.int2str[idx] = new_token
                self.merges.append(best_pair)

                # Merge every occurrence of best_pair in O(occurrences)
                positions = list(pair_positions.pop(best_pair, set()))
                pair_counts[best_pair] = 0

                for i in positions:
                    if tokens[i] is None:
                        continue
                    j = next_arr[i]
                    if j >= n or tokens[j] is None:
                        continue
                    if (tokens[i], tokens[j]) != best_pair:
                        continue  # position became stale (already merged)

                    pi = prev_arr[i]
                    nj = next_arr[j]

                    # Detach the left-neighbor pair that ends at i
                    if pi >= 0 and tokens[pi] is not None:
                        lp = (tokens[pi], tokens[i])
                        pair_counts[lp] -= 1
                        pair_positions[lp].discard(pi)

                    # Detach the right-neighbor pair that starts at j
                    if nj < n and tokens[nj] is not None:
                        rp = (tokens[j], tokens[nj])
                        pair_counts[rp] -= 1
                        pair_positions[rp].discard(j)

                    # Perform merge: overwrite i, tombstone j, relink list
                    tokens[i] = new_token
                    tokens[j] = None
                    next_arr[i] = nj
                    if nj < n:
                        prev_arr[nj] = i

                    # Register new left pair  (pi, i)
                    if pi >= 0 and tokens[pi] is not None:
                        lp = (tokens[pi], new_token)
                        pair_counts[lp] += 1
                        pair_positions[lp].add(pi)
                        heapq.heappush(heap, (-pair_counts[lp], lp))

                    # Register new right pair (i, nj)
                    if nj < n and tokens[nj] is not None:
                        rp = (new_token, tokens[nj])
                        pair_counts[rp] += 1
                        pair_positions[rp].add(i)
                        heapq.heappush(heap, (-pair_counts[rp], rp))

                pbar.update(1)
                pbar.set_postfix(vocab=len(self.str2int), freq=best_count)

        elapsed = time() - t0
        logger.info(
            f"Trained BPE tokenizer | vocab: {len(self.str2int)} "
            f"| merges: {len(self.merges)} | {elapsed:.2f}s"
        )

    def encode(self, sequence: str) -> torch.Tensor:
        """Encode *sequence* to a list of token IDs.

        Special tokens are matched atomically before BPE is applied to the
        remaining segments, so they are never split into subword pieces.
        """
        unk_id = self.str2int.get(self.unk_token, 1)
        ids: List[int] = []

        for chunk, is_special in self._split_on_special_tokens(sequence):
            if is_special:
                ids.append(self.str2int.get(chunk, unk_id))
                continue

            chunk_tokens: List[str] = list(chunk)
            for pair in self.merges:
                i = 0
                merged: List[str] = []
                while i < len(chunk_tokens):
                    if (
                        i < len(chunk_tokens) - 1
                        and chunk_tokens[i] == pair[0]
                        and chunk_tokens[i + 1] == pair[1]
                    ):
                        merged.append(pair[0] + pair[1])
                        i += 2
                    else:
                        merged.append(chunk_tokens[i])
                        i += 1
                chunk_tokens = merged

            ids.extend(self.str2int.get(tok, unk_id) for tok in chunk_tokens)

        return torch.tensor(ids)

    def decode(self, ids: Union[torch.Tensor, np.ndarray]) -> str:
        """Decode a list of token IDs back to the original string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.numpy()

        return "".join(self.int2str.get(int(i), self.unk_token or "<|UNK|>") for i in ids)


# ----------------------------------------------------------------------

if __name__ == "__main__":
    with open("dataset/The_Verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = BPETokenizer()
    tokenizer.fit(raw_text, vocab_size=452, min_freq=None)

    # ---- Print vocabulary ------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"{'TOKEN VOCABULARY':^60}")
    print(f"{'=' * 60}")
    for idx, token in tokenizer.int2str.items():
        # Show whitespace / control chars unambiguously
        display = repr(token) if any(c in token for c in (" ", "\n", "\t", "\r")) else token
        print(f"  [{idx:>4}]  {display}")

    # ---- Save tokenizer --------------------------------------------------
    # save_path = "dataset/tokenizer.json"
    # tokenizer.save(save_path)

    # ---- Reload and verify -----------------------------------------------
    # print(f"\n{'=' * 60}")
    # print("Reload check (restoring from file, no refit):")
    # loaded = BPETokenizer(vocab_file=save_path)

    # sample = raw_text[:300]
    # for tok, (orig_ids, load_ids) in {
    #     "encode": (tokenizer.encode(sample), loaded.encode(sample)),
    #     "decode": (tokenizer.decode(tokenizer.encode(sample)),
    #                loaded.decode(loaded.encode(sample))),
    # }.items():
    #     match = orig_ids == load_ids if tok == "encode" else orig_ids == load_ids
    #     print(f"  {tok} matches original : {match}")

    # # ---- Round-trip sanity check -----------------------------------------
    # ids = loaded.encode(sample)
    # decoded = loaded.decode(ids)

    # print(f"\n{'=' * 60}")
    # print("Round-trip check (first 300 chars of training text):")
    # print(f"  Lossless : {sample == decoded}")
    # print(f"  Original : {sample[:80]!r}")
    # print(f"  Decoded  : {decoded[:80]!r}")
    # print(f"  Tokens   : {len(ids)} ids for {len(sample)} chars  "
    #       f"(compression {len(sample)/len(ids):.2f}x)")
    # print(f"  IDs[0:20]: {ids[:20]}")
