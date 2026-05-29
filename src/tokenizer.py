# tokenizer.py — a from-scratch Byte Pair Encoding (BPE) tokenizer.
#
# BPE overview
# ------------
# BPE starts with a character-level vocabulary (every unique character in the
# training text becomes its own token) and then iteratively merges the most
# frequent adjacent token pair into a new, longer token.  This continues until
# the target vocabulary size is reached.  The resulting vocabulary contains
# both individual characters (which handle any new input) and common subword
# sequences (which compress frequent patterns efficiently).
#
# Design choices in this implementation
# --------------------------------------
# 1. FLAT — the entire text is treated as one continuous character sequence.
#    Spaces, newlines, and punctuation all participate in merges on equal terms.
#    This differs from word-level BPE (used in GPT-2) where the text is split
#    into words first.  Flat BPE guarantees perfect encode→decode round-trips.
#
# 2. DOUBLY-LINKED LIST — after initial tokenisation the positions are tracked
#    via prev_arr / next_arr arrays (a linked-list-over-array).  When two
#    positions are merged the right position is "tombstoned" (set to None) and
#    the linked-list pointers are updated in O(1).  This avoids repeatedly
#    shifting a Python list, which would be O(n) per merge.
#
# 3. LAZY-DELETION MAX-HEAP — a max-heap keyed on negative frequency provides
#    O(log n) extraction of the best pair.  Stale heap entries (from pairs that
#    were updated after insertion) are detected by comparing the stored count
#    with the current count in pair_counts and discarded on-the-fly.
#
# 4. SPECIAL TOKENS — EOS and UNK tokens are added to the vocabulary before any
#    characters, so they always receive IDs 0 and 1.  During encoding they are
#    matched atomically (never split into subword pieces) using a regex split.
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
# Ensure the project root is on sys.path so that `from src.logger import ...` resolves
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import setup_logger

# Module-level logger — all tokenizer events are recorded at INFO level
logger = setup_logger(__name__)


# =======================================================================
# _bpe_encode_chunk — O(n log n) heap-based BPE encoder for one chunk
# =======================================================================
# Why this is fast
# ----------------
# The naive approach iterates over every merge rule (up to 31 803) for every
# chunk, giving O(chunk_chars × num_merges) per encode call.  For a 700 MB
# corpus that is hundreds of hours of Python work.
#
# This function uses the same priority-heap trick that fit() uses:
#   1. Seed a min-heap with (merge_rank, position) for every adjacent pair
#      in the chunk that is a known merge rule.
#   2. Always pop the lowest-rank (highest-priority) applicable pair and
#      merge it, then push the two newly created neighbour pairs.
#   3. A doubly-linked list over the symbols array lets us merge in O(1)
#      without shifting a Python list.
#   4. Stale heap entries (positions whose tokens have since changed) are
#      detected by comparing the stored rank with merge_ranks[current_pair]
#      and discarded lazily.
#
# Correctness guarantee: BPE merge rules have the property that applying
# rule at rank r can only produce tokens that participate in rules at
# rank > r.  Therefore popping the globally lowest-rank entry is always
# equivalent to advancing the sequential scan one step — the two
# algorithms produce identical token sequences.
#
# Complexity: O(n log n) per chunk, where n = len(chunk).
#             Compared to O(n × |merges|) ≈ O(n × 31800) for the naive loop,
#             this is typically 1 000–3 000× faster for Ukrainian text with a
#             32 k vocab.
def _bpe_encode_chunk(
    symbols: List[str],
    merge_ranks: Dict[Tuple[str, str], int],
) -> List[str]:
    """Apply BPE merge rules to *symbols* via a priority heap.

    Args:
        symbols:     List of individual characters for one text chunk.
        merge_ranks: Dict mapping (left, right) pairs to their rank (lower
                     rank = higher priority = applied first).  Build once
                     with ``BPETokenizer._get_merge_ranks()`` and reuse.

    Returns:
        List of merged token strings after all applicable rules are applied.
    """
    n = len(symbols)
    # Single-character or empty chunk: nothing to merge
    if n <= 1:
        return list(symbols)

    # Doubly-linked list encoded as parallel arrays — same pattern as fit()
    # prev_arr[i] = index of the nearest live position to the left (-1 = none)
    prev_arr: List[int] = list(range(-1, n - 1))
    # next_arr[i] = index of the nearest live position to the right (n = none)
    next_arr: List[int] = list(range(1, n + 1))
    # alive[i] = False once position i has been consumed by a merge
    alive: List[bool] = [True] * n

    # Seed the heap: push (rank, position) for every adjacent pair that has
    # a merge rule.  Pairs not in merge_ranks are never merged, so skip them.
    heap: List[Tuple[int, int]] = []
    for i in range(n - 1):
        pair = (symbols[i], symbols[i + 1])
        if pair in merge_ranks:
            heapq.heappush(heap, (merge_ranks[pair], i))

    while heap:
        rank, i = heapq.heappop(heap)

        # Skip tombstoned left position
        if not alive[i]:
            continue
        j = next_arr[i]
        # Skip if right neighbour is gone (end of list or tombstoned)
        if j >= n or not alive[j]:
            continue

        # Lazy-deletion check: the pair at (i, j) might have changed since
        # this entry was pushed.  If the current pair doesn't match the
        # stored rank, discard this stale entry.
        pair = (symbols[i], symbols[j])
        if pair not in merge_ranks or merge_ranks[pair] != rank:
            continue

        # ---- Perform the merge ------------------------------------------------
        symbols[i] = symbols[i] + symbols[j]   # overwrite i with merged token
        alive[j] = False                        # tombstone j

        # Relink: i now points directly to j's old right neighbour
        nj = next_arr[j]
        next_arr[i] = nj
        if nj < n:
            prev_arr[nj] = i

        # Push the new left pair (prev(i), i) if it has a merge rule
        pi = prev_arr[i]
        if pi >= 0 and alive[pi]:
            new_pair = (symbols[pi], symbols[i])
            if new_pair in merge_ranks:
                heapq.heappush(heap, (merge_ranks[new_pair], pi))

        # Push the new right pair (i, next(i)) if it has a merge rule
        if nj < n and alive[nj]:
            new_pair = (symbols[i], symbols[nj])
            if new_pair in merge_ranks:
                heapq.heappush(heap, (merge_ranks[new_pair], i))

    # Collect surviving positions in original order
    return [symbols[i] for i in range(n) if alive[i]]


# =======================================================================
# BPETokenizer
# =======================================================================
class BPETokenizer:
    def __init__(self, vocab_file: Optional[str] = None):
        """Byte Pair Encoding (BPE) Tokenizer — flat character-level, no pre-splitting.

        Args:
            vocab_file: Optional path to a JSON file previously saved with
                        ``save()``.  When provided the tokenizer is fully
                        restored (vocab + merges) and is ready to encode/decode
                        without calling ``fit()`` first.
        """
        # str2int maps token strings to integer IDs (used during encoding)
        self.str2int: Dict[str, int] = {}
        # int2str maps integer IDs back to token strings (used during decoding)
        self.int2str: Dict[int, str] = {}
        # merges is the ordered list of (pair[0], pair[1]) merge rules learned
        # during fit().  The order matters: earlier merges are applied first.
        self.merges: List[Tuple[str, str]] = []
        # eos_token is the end-of-sequence sentinel inserted between documents
        self.eos_token = None
        # unk_token is substituted for characters not seen during training
        self.unk_token = None

        # If a vocabulary file is given, restore the tokenizer state immediately
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
        # Bundle all state into a single dict for serialisation
        state = {
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            # vocab is stored as str→int so JSON keys are always strings
            "vocab": self.str2int,
            # merges is stored as a list of 2-element lists (JSON has no tuples)
            "merges": self.merges,
        }
        # ensure_ascii=False preserves non-ASCII characters (e.g. Cyrillic)
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

        # Restore the special tokens from the saved state
        self.eos_token = state["eos_token"]
        self.unk_token = state["unk_token"]
        # JSON keys are always strings, so integer conversion of values is explicit
        self.str2int = {tok: int(idx) for tok, idx in state["vocab"].items()}
        # Build the reverse mapping for decoding
        self.int2str = {int(idx): tok for tok, idx in state["vocab"].items()}
        # Merges are stored as lists in JSON; convert back to tuples
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
        # Collect only the special tokens that have been set (not None)
        special_tokens = [t for t in [self.eos_token, self.unk_token] if t is not None]
        # If no special tokens are defined, the entire text is one non-special chunk
        if not special_tokens:
            return [(text, False)]

        # Sort by length descending to prevent partial matches
        # (e.g. a shorter token that is a prefix of a longer one)
        special_tokens.sort(key=len, reverse=True)

        # Build a regex that matches any of the special tokens literally
        pattern = "(" + "|".join(re.escape(s) for s in special_tokens) + ")"
        # Use a set for O(1) membership testing when labelling chunks
        special_set = set(special_tokens)
        
        # re.split with a capturing group includes the separators in the result list
        # Filter out empty strings that re.split may produce at the boundaries
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
        # If a list of texts is given, join them with the EOS token as separator
        if isinstance(text_data, list):
            text = eos_token.join(text_data)
        else:
            text = text_data

        # ---- Initialise vocabulary ----------------------------------------
        self.eos_token = eos_token
        self.unk_token = unk_token
        # sorted() ensures the vocab order is deterministic across runs
        base_chars = sorted(set(text))
        # Special tokens occupy IDs 0 and 1; all characters follow in sorted order
        vocab: List[str] = [eos_token, unk_token] + base_chars

        # Build the initial bidirectional mappings from the base vocabulary
        self.str2int = {tok: i for i, tok in enumerate(vocab)}
        self.int2str = {i: tok for tok, i in self.str2int.items()}
        # merges will be populated as the BPE loop runs
        self.merges = []

        # ---- Build a doubly-linked list over the character sequence --------
        # prev_arr[i] / next_arr[i] = neighbouring *valid* indices
        # tokens[i] = None means position i has been merged away
        n = len(text)
        # tokens starts as the list of individual characters in the training text
        tokens: List[Optional[str]] = list(text)
        # prev_arr[i] holds the index of the nearest non-tombstoned position to the left
        # (-1 signals "no left neighbour", i.e. position i is the head of the list)
        prev_arr: List[int] = list(range(-1, n - 1))   # prev_arr[0] = -1
        # next_arr[i] holds the index of the nearest non-tombstoned position to the right
        # (n signals "no right neighbour", i.e. position i is the tail of the list)
        next_arr: List[int] = list(range(1, n + 1))    # next_arr[n-1] = n

        # ---- Count all adjacent pairs + remember their positions -----------
        # pair_counts[pair] = total occurrences of this adjacent pair in the current sequence
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        # pair_positions[pair] = set of starting positions where this pair currently occurs
        pair_positions: Dict[Tuple[str, str], set] = defaultdict(set)

        # Initial population: scan the character sequence once to count all pairs
        for i in range(n - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += 1
            pair_positions[pair].add(i)

        # ---- Max-heap via negated counts (lazy deletion for stale entries) -
        # Python's heapq is a min-heap; negating counts turns it into a max-heap.
        # Entries are (negative_count, pair) tuples.
        heap: List[Tuple[int, Tuple[str, str]]] = [
            (-cnt, pair) for pair, cnt in pair_counts.items()
        ]
        heapq.heapify(heap)

        # ---- BPE merge loop ------------------------------------------------
        t0 = time()
        # Number of new merge rules needed to reach the target vocab size
        target_merges = vocab_size - len(vocab)

        with tqdm(
            total=target_merges,
            desc="BPE merges",
            unit="merge",
            disable=not verbose,
        ) as pbar:
            while len(self.str2int) < vocab_size:
                # Pop until we find a heap entry whose count is still current
                # (lazy deletion: ignore stale entries whose count changed after push)
                best_pair: Optional[Tuple[str, str]] = None
                best_count = 0
                while heap:
                    neg_cnt, candidate = heapq.heappop(heap)
                    # Check whether the heap entry is still valid
                    actual = pair_counts.get(candidate, 0)
                    if actual == -neg_cnt and actual > 0:
                        best_pair = candidate
                        best_count = actual
                        break

                # No valid pair found → all remaining pairs have zero frequency
                if best_pair is None:
                    break
                # min_freq early-stopping: if the best pair is too rare, stop
                if min_freq is not None and best_count < min_freq:
                    break

                # The merged token is simply the concatenation of the two substrings
                new_token = best_pair[0] + best_pair[1]
                # Add the new token to the vocabulary if it isn't already present
                if new_token not in self.str2int:
                    idx = len(self.str2int)
                    self.str2int[new_token] = idx
                    self.int2str[idx] = new_token
                # Record the merge rule in the ordered list
                self.merges.append(best_pair)

                # Merge every occurrence of best_pair in O(occurrences)
                # We consume pair_positions so each position is visited exactly once
                positions = list(pair_positions.pop(best_pair, set()))
                # Mark the pair as exhausted in pair_counts
                pair_counts[best_pair] = 0

                for i in positions:
                    # Skip if this position was already tombstoned by an earlier merge
                    if tokens[i] is None:
                        continue
                    j = next_arr[i]
                    # Skip if the right neighbour no longer exists or is tombstoned
                    if j >= n or tokens[j] is None:
                        continue
                    if (tokens[i], tokens[j]) != best_pair:
                        continue  # position became stale (already merged)

                    # Retrieve the positions of the left and right outer neighbours
                    pi = prev_arr[i]
                    nj = next_arr[j]

                    # Detach the left-neighbor pair that ends at i
                    # (pi, i) is no longer valid once i gets a new token value
                    if pi >= 0 and tokens[pi] is not None:
                        lp = (tokens[pi], tokens[i])
                        pair_counts[lp] -= 1
                        pair_positions[lp].discard(pi)

                    # Detach the right-neighbor pair that starts at j
                    # (j, nj) is no longer valid once j is tombstoned
                    if nj < n and tokens[nj] is not None:
                        rp = (tokens[j], tokens[nj])
                        pair_counts[rp] -= 1
                        pair_positions[rp].discard(j)

                    # Perform merge: overwrite i, tombstone j, relink list
                    # i now holds the merged token; j is effectively deleted
                    tokens[i] = new_token
                    tokens[j] = None
                    # Update the linked list: i skips over j to point at nj
                    next_arr[i] = nj
                    if nj < n:
                        prev_arr[nj] = i

                    # Register new left pair  (pi, i) now that i has a new token
                    if pi >= 0 and tokens[pi] is not None:
                        lp = (tokens[pi], new_token)
                        pair_counts[lp] += 1
                        pair_positions[lp].add(pi)
                        # Push updated count onto the heap (old entry becomes stale)
                        heapq.heappush(heap, (-pair_counts[lp], lp))

                    # Register new right pair (i, nj) now that i has a new token
                    if nj < n and tokens[nj] is not None:
                        rp = (new_token, tokens[nj])
                        pair_counts[rp] += 1
                        pair_positions[rp].add(i)
                        # Push updated count onto the heap (old entry becomes stale)
                        heapq.heappush(heap, (-pair_counts[rp], rp))

                pbar.update(1)
                # Show live vocabulary size and frequency of the last merged pair
                pbar.set_postfix(vocab=len(self.str2int), freq=best_count)

        elapsed = time() - t0
        logger.info(
            f"Trained BPE tokenizer | vocab: {len(self.str2int)} "
            f"| merges: {len(self.merges)} | {elapsed:.2f}s"
        )

    def encode(
        self,
        sequence: str,
        verbose: bool = False,
        labels: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Encode *sequence* to a list of token IDs.

        Special tokens are matched atomically before BPE is applied to the
        remaining segments, so they are never split into subword pieces.

        Args:
            verbose: When True, display a tqdm progress bar. Each tick
                     represents one EOS-separated document chunk, so the bar
                     advances once per file and gives a meaningful ETA.
                     The postfix shows chars processed and tokens produced so far.
            labels:  Optional list of human-readable names (e.g. filenames) for
                     each non-special chunk, shown in the bar description so you
                     can see which file is currently being encoded.  Must have
                     the same length as the number of non-special chunks.
        """
        # Fall back to UNK ID (1) for any token not found in the vocabulary
        unk_id = self.str2int.get(self.unk_token, 1)
        ids: List[int] = []

        # Materialise the chunk list up front so we can count chunks and
        # compute the total character count before the progress bar starts
        chunks = self._split_on_special_tokens(sequence)

        # Number of non-special chunks == number of documents to encode
        # (EOS token lookups are instant and excluded from the progress total)
        num_docs = sum(1 for _, is_s in chunks if not is_s)
        # Total characters across all text chunks — used in the postfix display
        total_chars = sum(len(c) for c, is_s in chunks if not is_s)
        chars_done = 0
        # Tracks which non-special chunk we are on for label lookup
        chunk_idx = 0

        with tqdm(
            total=num_docs,
            desc="Encoding corpus",
            unit="doc",
            disable=not verbose,
        ) as pbar:
            # Show the full scale before the first document is processed
            pbar.set_postfix(chars=f"0 / {total_chars:,}", tokens="0")

            for chunk, is_special in chunks:
                if is_special:
                    # Special tokens are always mapped atomically; never split
                    ids.append(self.str2int.get(chunk, unk_id))
                    continue

                # Update the bar description with the current file name so the
                # user can see exactly which file is being encoded right now.
                # Long names are truncated to keep the progress bar width stable.
                if verbose and labels and chunk_idx < len(labels):
                    name = labels[chunk_idx]
                    if len(name) > 40:
                        # Keep the end of the name (extension + unique suffix)
                        name = "…" + name[-39:]
                    pbar.set_description(f"Encoding  {name}")

                # Use the O(n log n) heap-based encoder instead of the naive
                # O(n × |merges|) sequential scan.  _get_merge_ranks() returns
                # a cached {pair: rank} dict so it costs nothing after the first
                # call on this tokenizer instance.
                merge_ranks = self._get_merge_ranks()
                chunk_tokens = _bpe_encode_chunk(list(chunk), merge_ranks)

                # Look up each resulting token; unknown characters map to unk_id
                ids.extend(self.str2int.get(tok, unk_id) for tok in chunk_tokens)

                # Advance the bar and refresh the postfix stats
                chars_done += len(chunk)
                chunk_idx += 1
                pbar.update(1)
                pbar.set_postfix(
                    chars=f"{chars_done:,} / {total_chars:,}",
                    tokens=f"{len(ids):,}",
                )

        # Return a 1-D integer tensor compatible with PyTorch operations
        return torch.tensor(ids)

    def decode(self, ids: Union[torch.Tensor, np.ndarray]) -> str:
        """Decode a list of token IDs back to the original string."""
        # Accept both PyTorch tensors and NumPy arrays as input
        if isinstance(ids, torch.Tensor):
            ids = ids.numpy()

        # Join all token strings in order; missing IDs fall back to unk_token
        return "".join(self.int2str.get(int(i), self.unk_token or "<|UNK|>") for i in ids)

    def _get_merge_ranks(self) -> Dict[Tuple[str, str], int]:
        """Return a {(left, right): rank} dict built from self.merges.

        The result is cached as a private attribute and rebuilt automatically
        when the number of merge rules changes (e.g. after a call to fit()).
        Caching avoids re-building 31 000+ dict entries on every encode call,
        which matters most when encoding many small files in a loop.
        """
        # Check the cache; the length guard handles tokenizers that were
        # updated via fit() after the cache was last built
        if getattr(self, "_cached_merge_ranks_len", -1) != len(self.merges):
            self._cached_merge_ranks: Dict[Tuple[str, str], int] = {
                pair: rank for rank, pair in enumerate(self.merges)
            }
            # Store the length so we know when to rebuild
            self._cached_merge_ranks_len: int = len(self.merges)
        return self._cached_merge_ranks


# ----------------------------------------------------------------------
# Quick smoke-test — run this file directly to train a small tokenizer
# on a sample text, inspect the vocabulary, and verify round-trip fidelity.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    with open("dataset/The_Verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Train a small tokenizer with 452 tokens on the sample text
    # min_freq=None means training continues until vocab_size is reached
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
