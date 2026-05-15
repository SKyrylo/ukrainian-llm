import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    """Sliding-window dataset built from pre-tokenized token IDs.

    Accepts a 1-D integer Tensor produced by ``tokenizer.encode()``.
    Keeping tokenization separate from the dataset allows token IDs to be
    cached to disk and reused across runs without re-encoding the corpus.
    """

    def __init__(self, token_ids: Tensor, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(token_ids[i : i + max_length].clone())
            self.target_ids.append(token_ids[i + 1 : i + max_length + 1].clone())

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    token_ids: Tensor,
    batch_size: int = 4,
    context_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader from pre-tokenized token IDs."""
    dataset = CustomDataset(token_ids, context_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.tokenizer import BPETokenizer

    with open("dataset/The_Verdict.txt") as f:
        raw_text = f.read()
    
    tokenizer = BPETokenizer("dataset/tokenizer.json")
    dataloader = create_dataloader(raw_text, tokenizer, batch_size=8, max_length=4, stride=4, shuffle=False)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print(f"Token IDs: {inputs}")
    print(f"Input shape: {inputs.shape}")
    
    print(tokenizer.decode(inputs))
