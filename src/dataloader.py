# dataloader.py — constructs PyTorch Dataset and DataLoader objects for
# language-model training from raw tokenised text.
#
# The sliding-window approach used here creates (input, target) pairs where
# the target is the input shifted one position to the right.  This is the
# standard next-token-prediction setup used to train GPT-style models.
import torch
from torch.utils.data import Dataset, DataLoader


# CustomDataset converts a single long string of text into a collection of
# fixed-length overlapping windows suitable for batch training.
# Each window of length `max_length` becomes one training sample whose target
# is the same window shifted right by one token (next-token prediction).
class CustomDataset(Dataset):
    # txt       : the full training or validation text as a single string
    # tokenizer : a fitted BPETokenizer instance used to encode the text
    # max_length: the context window size — must match the model's context_length
    # stride    : how many tokens to advance between consecutive windows;
    #             stride < max_length creates overlapping windows (more data),
    #             stride == max_length gives non-overlapping windows (less data)
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Encode the entire text up front; result is a 1-D integer tensor
        token_ids = tokenizer.encode(txt)

        # Slide a window of size max_length over the token sequence.
        # The loop stops before the last window would lack a full target chunk.
        for i in range(0, len(token_ids) - max_length, stride):
            # Input chunk: tokens at positions [i, i+max_length)
            input_chunk  = token_ids[i:i + max_length]
            # Target chunk: same window shifted one step to the right,
            # so target[t] is the token that should follow input[t]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            # .clone() avoids sharing storage with the original tensor
            self.input_ids.append(input_chunk.clone())
            self.target_ids.append(target_chunk.clone())

    # Returns the total number of windows (training samples) in the dataset
    def __len__(self):
        return len(self.input_ids)

    # Returns the (input_ids, target_ids) pair for the window at position idx
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# create_dataloader is the public factory function that wraps CustomDataset
# inside a PyTorch DataLoader for batched, optionally shuffled iteration.
#
# Parameters
# ----------
# txt            : raw text string (training split or validation split)
# tokenizer      : fitted BPETokenizer
# batch_size     : number of windows per mini-batch
# context_length : window size passed through to CustomDataset as max_length
# stride         : step size between windows (use context_length for no overlap)
# shuffle        : True during training; False during validation to keep order
# drop_last      : True during training to guarantee uniform batch sizes;
#                  False during validation to evaluate on every sample
# num_workers    : number of parallel worker processes for data loading
def create_dataloader(
    txt,
    tokenizer,
    batch_size=4,
    context_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    # Tokenise and window the text into a CustomDataset first
    dataset = CustomDataset(txt, tokenizer, context_length, stride)
    # Wrap the dataset in a DataLoader for batched iteration during training
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


# -----------------------------------------------------------------------
# Quick smoke-test: run this file directly to verify the dataloader works
# with a real tokenizer and text file.
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    # Add the project root to sys.path so that `src.*` imports resolve correctly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.tokenizer import BPETokenizer

    with open("dataset/The_Verdict.txt") as f:
        raw_text = f.read()

    # Load a pre-trained tokenizer from disk (must already exist)
    tokenizer = BPETokenizer("dataset/tokenizer.json")
    # Create a small dataloader with context_length=4 for easy inspection
    dataloader = create_dataloader(raw_text, tokenizer, batch_size=8, context_length=4, stride=4, shuffle=False)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    # Print raw token IDs and decoded text to verify correctness
    print(f"Token IDs: {inputs}")
    print(f"Input shape: {inputs.shape}")
    print(tokenizer.decode(inputs))
