import pandas as pd
from datasets import load_dataset
from itertools import islice
from sentence_transformers import SentenceTransformer


# USE STREAMING MODE
fine_web_ds = load_dataset(
    "HuggingFaceFW/fineweb-edu-llama3-annotations", split="train", streaming=True
)

# Take only the first 1000 samples
num_samples = 100
sampled_data = list(islice(fine_web_ds, num_samples))

data = pd.DataFrame(sampled_data)
data.drop(["metadata", "prompt"], inplace=True, axis=1)

data.to_parquet("data/tiny-dataset.parquet")


def one_hot_encode_score(score, num_classes=6):
    one_hot = [0] * num_classes
    one_hot[score] = 1
    return one_hot


# Apply the one-hot encoding to the score column
data["score"] = data["score"].apply(lambda x: one_hot_encode_score(x, num_classes=6))

# Now, let's embed the text using the model
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-s")

batch_size = 128  # Adjust batch size based on your GPU/CPU memory
texts = data["text"].tolist()
embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

data["embedding"] = embeddings.tolist()
tmp_text = data["text"].tolist()[:10]
tmp_emb = data["embedding"].tolist()[:10]
data.drop("text", inplace=True, axis=1)

print(data.head())

data.to_csv("data/tiny-dataset-processed.csv")
