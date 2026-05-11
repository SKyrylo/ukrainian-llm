import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(
        self,
        txt,
        tokenizer,
        max_length,
        stride
        ):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)

        # Use the sliding window to chunk the data into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]

            self.input_ids.append(input_chunk.clone())
            self.target_ids.append(target_chunk.clone())

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    txt,
    tokenizer,
    batch_size=4,
    context_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
    ):
    # Create the dataset
    dataset = CustomDataset(txt, tokenizer, context_length, stride)

    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


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
