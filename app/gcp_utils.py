from google.cloud import storage
import pandas as pd
import torch
import os
import tempfile


def upload_to_bucket(
    bucket_name: str, source_file_path: str, destination_blob_name: str
):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_path)


def load_model_from_gcp(bucket_name: str, model_path: str):
    """
    Load a TorchScript model from a GCP bucket.

    Args:
        bucket_name (str): Name of the GCP bucket.
        model_path (str): Path to the model file in the bucket.

    Returns:
        torch.jit.ScriptModule: The loaded TorchScript model.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_path)

    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        blob.download_to_filename(temp_file.name)
        model = torch.jit.load(temp_file.name)

    return model


def fetch_input_from_bucket(bucket_name: str, data_path: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(data_path)

    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        blob.download_to_filename(temp_file.name)
        return pd.read_parquet(temp_file.name)


def upload_directory_to_bucket(
    bucket_name: str, source_directory: str, destination_prefix: str = ""
):
    """
    Uploads all files in a directory to a GCP bucket.

    Args:
        bucket_name (str): The name of the GCP bucket.
        source_directory (str): Path to the local directory to upload.
        destination_prefix (str): Path prefix in the bucket (optional). If provided, files will be uploaded under this path.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(source_directory):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, source_directory)
            destination_blob_name = os.path.join(destination_prefix, relative_path)

            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(local_file_path)


def fetch_directory_from_bucket(
    bucket_name: str, source_prefix: str, destination_directory: str
):
    """
    Fetches all files from a directory (prefix) in a GCP bucket and downloads them locally.

    Args:
        bucket_name (str): The name of the GCP bucket.
        source_prefix (str): Path prefix in the bucket to fetch files from (e.g., "datasets/parquet").
        destination_directory (str): Local directory to save the downloaded files.
    """
    client = storage.Client()
    os.makedirs(destination_directory, exist_ok=True)

    blobs = client.list_blobs(bucket_name)
    for blob in blobs:
        relative_path = os.path.relpath(blob.name, source_prefix)
        local_file_path = os.path.join(destination_directory, relative_path)

        os.makedirs(os.path.dirname((local_file_path)), exist_ok=True)
        blob.download_to_filename(local_file_path)


if __name__ == "__main__":
    upload_to_bucket("fineweb-datasets", "test.txt", "test.txt")
    # upload_directory_to_bucket(
    #     "fineweb-datasets",
    #     "data/tiny-dataset-processed.parquet",
    #     "tiny-dataset-processed",
    # )
