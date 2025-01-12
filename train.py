from model import Classifier
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from utils import prepare_dataloaders

model = Classifier(384, 256, 6)

n_epochs = 10
batch_size = 32
learning_rate = 0.001


train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders()

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
            predicted = torch.argmax(y_pred.data, 1)
            y = torch.argmax(y, 1)
            total += y.size(0)
            correct += (predicted == y).sum()

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}")
