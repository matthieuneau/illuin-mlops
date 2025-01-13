from gcp_utils import upload_to_bucket
from model import Classifier
import torch
from utils import prepare_dataloaders

model = Classifier(384, 256, 6)

n_epochs = 30
batch_size = 32
learning_rate = 0.0005


train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(
    "data/tiny-dataset-processed.parquet"
)

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


scripted_model = torch.jit.script(model)
scripted_model.save("models/classifier.pt")

upload_to_bucket("fineweb-classifiers", "models/classifier.pt", "classifier.pt")
