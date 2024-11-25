import pandas as pd

data = pd.read_csv("tiny-dataset.csv")
data.drop(["metadata", "prompt"], inplace=True, axis=1)
# data = data.iloc[:100]

# print(data.head())

# Now, let's embed the text using the model
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Snowflake/snowflake-arctic-embed-s")

batch_size = 128  # Adjust batch size based on your GPU/CPU memory
texts = data["text"].tolist()
embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

data["embedding"] = embeddings.tolist()
data.drop("text", inplace=True, axis=1)

print(data.head())

data.to_parquet("datasets/tiny-dataset.parquet")
