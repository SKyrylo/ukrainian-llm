import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import BPETokenizer


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

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
    ):
    # Initialize the tokenizer
    tokenizer = BPETokenizer("dataset/tokenizer.json")

    # Create the dataset
    dataset = CustomDataset(txt, tokenizer, max_length, stride)

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
    with open("dataset/The_Verdict.txt") as f:
        raw_text = f.read()
    
    dataloader = create_dataloader(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    tokenizer = BPETokenizer("dataset/tokenizer.json")
    
    print(tokenizer.decode(first_batch[0]))
    print(tokenizer.decode(first_batch[1]))
