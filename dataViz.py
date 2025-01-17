from utils import prepare_dataloaders
import matplotlib.pyplot as plt
from collections import Counter
import torch


train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders()


def analyze_labels(dataloader, split_name):
    all_labels = []

    # Iterate through the dataloader to collect labels
    for _, labels in dataloader:
        # Convert one-hot encoded vectors to class indices
        class_indices = torch.argmax(labels, dim=1).tolist()
        all_labels.extend(class_indices)

    # Count the occurrences of each class
    label_counts = Counter(all_labels)

    # Sort the counts by class index for visualization
    sorted_counts = dict(sorted(label_counts.items()))
    classes, counts = zip(*sorted_counts.items())

    plt.figure(figsize=(8, 5))
    plt.bar(classes, counts, tick_label=[cls for cls in classes])
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(f"{split_name} Label Distribution")
    plt.show()


analyze_labels(train_dataloader, "Training")
