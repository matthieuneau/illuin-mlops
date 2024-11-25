import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

batch_size = 32


class MyDataset(Dataset):
    def __init__(self, embedding, score):
        self.embedding = embedding
        self.score = score

    def __len__(self):
        return len(self.score)

    def __getitem__(self, idx):
        x = self.embedding[idx]
        y = self.score[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


def prepare_dataloaders():
    df = pd.read_parquet("datasets/tiny-dataset.parquet")

    embeddings = df["embedding"].values
    score = df["score"].values

    dataset = MyDataset(embeddings, score)
    total_size = len(dataset)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
