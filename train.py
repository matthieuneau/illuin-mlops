from model import Classifier
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

model = Classifier(384, 256, 5)

n_epochs = 10
batch_size = 32
learning_rate = 0.001


class MyDataset(Dataset):
    def __init__(self, embedding, score):
        self.embedding = embedding
        self.score = score

    def __len__(self):
        return len(self.score)

    def __getitem__(self, idx):
        x = self.embedding[idx]
        y = self.score[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)


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

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    model.train()
    for i, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (x, y) in enumerate(val_dataloader):
            y_pred = model(x)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}")
