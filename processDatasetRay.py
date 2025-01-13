"""
redundant with what the second part of build dataset does. it's to test ray data
"""

import ray
from sentence_transformers import SentenceTransformer
from gcp_utils import upload_to_bucket
import numpy as np

ds = ray.data.read_parquet("data/tiny-dataset.parquet")

# Scores range from 0 to 5 included
N_CLASSES = 6


def one_hot_encode_score(row):
    score = row["score"]
    one_hot = np.zeros(N_CLASSES, dtype=np.int32)
    one_hot[score] = 1
    row["score"] = one_hot
    return row


ds = ds.map(one_hot_encode_score)

ds.show(limit=1)


# Performed on CPU for now. Code can be changed to run on GPU
class DataEmbedding:
    def __init__(self) -> None:
        self.model = SentenceTransformer("Snowflake/snowflake-arctic-embed-s")
        self.model.eval()

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        texts = [str(text) for text in batch["text"]]
        embeddings = self.model.encode(texts, batch_size=128, show_progress_bar=False)
        batch["embedding"] = embeddings
        return batch


# Ray will automatically scale the nb of workers
ds = ds.map_batches(DataEmbedding, concurrency=(2, 6))

ds.write_parquet("data/tiny-dataset-processed.parquet")
