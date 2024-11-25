import pandas as pd

data = pd.read_csv("tiny-dataset.csv")
data.drop(["metadata", "prompt"], inplace=True, axis=1)

# Now, let's embed the text using the model
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Snowflake/snowflake-arctic-embed-s")

batch_size = 32  # Adjust batch size based on your GPU/CPU memory
texts = data["text"].tolist()
embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

data["embedding"] = embeddings.tolist()
tmp_text = data["text"].tolist()[:10]
tmp_emb = data["embedding"].tolist()[:10]
data.drop("text", inplace=True, axis=1)

print(data.head())

data.to_parquet("datasets/tiny-dataset.parquet")
