import glob
import os

import datasets
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

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


def prepare_dataloaders(data_path: str):
    parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
    df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)

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


def download_texts_from_hf(
    hf_dataset: str, n_texts: int, output_dir: str = "data/texts"
):
    """Download a specified number of text samples from a Hugging Face dataset and save them to a local directory.
    Args:
    hf_dataset (str): The name of the Hugging Face dataset to download from.
    n_texts (int): The number of text samples to download.
    output_dir (str): The local directory where the text samples will be saved.
    """

    dataset = datasets.load_dataset(
        hf_dataset,
        split="train",
        streaming=True,
    )
    dataset = dataset.take(n_texts)
    dataset = list(dataset)

    for i, text in enumerate(dataset):
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{i}.txt", "w") as f:
            f.write(text["text"])


if __name__ == "__main__":
    download_texts_from_hf(
        hf_dataset="Fishfishfishfishfish/Synthetic_text.txt",
        n_texts=1000,
        output_dir="data/texts",
    )
