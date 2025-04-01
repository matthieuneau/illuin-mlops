import torch
from gcp_utils import fetch_input_from_bucket, load_model_from_gcp


def predict(model_bucket: str, model_path: str, data_bucket: str, data_path: str):
    model = load_model_from_gcp(model_bucket, model_path)
    model.eval()

    input = fetch_input_from_bucket(data_bucket, data_path)

    embeddings = torch.tensor(input["embedding"].tolist(), dtype=torch.float32)

    with torch.no_grad():
        predictions = model(embeddings)

    return predictions.numpy().tolist()  # Convert tensor to list for JSON serialization


if __name__ == "__main__":
    print(
        predict(
            "fineweb-classifiers",
            "classifier.pt",
            "fineweb-datasets",
            "tiny-dataset-processed/3_000000_000000.parquet",
        )
    )
