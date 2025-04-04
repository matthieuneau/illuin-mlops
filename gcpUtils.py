import os
import tempfile

import pandas as pd
import torch
from google.cloud import aiplatform, storage


def upload_to_bucket(
    bucket_name: str, source_file_path: str, destination_blob_name: str
):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_path)


def load_model_from_bucket(bucket_name: str):
    """
    Load a TorchScript model from a GCP bucket. A bucket stores a single model in a model.pt file.
    Args:
        bucket_name (str): Name of the GCP bucket.
    Returns:
        torch.jit.ScriptModule: The loaded TorchScript model.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob("model.pt")

    # Create a named temporary file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        blob.download_to_filename(temp_path)
        model = torch.jit.load(temp_path)
        return model

    finally:
        # Clean up the temporary file
        import os

        if os.path.exists(temp_path):
            os.remove(temp_path)


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


def upload_gcs_model_to_vertex(
    gcs_model_path,
    display_name,
    project_id="swift-habitat-447619-k8",
    location="europe-west1",
    description=None,
    labels=None,
):
    """
    Upload a PyTorch model from a GCS bucket to Vertex AI Model Registry.

    Args:
        gcs_model_path (str): GCS path to the model file, e.g., 'gs://bucket-name/path/to/model.pt'
        display_name (str): Name to display in the Vertex AI Model Registry
        project_id (str): GCP project ID
        location (str): GCP region (default: 'us-central1')
        description (str, optional): Model description
        labels (dict, optional): Labels to attach to the model

    Returns:
        google.cloud.aiplatform.Model: The uploaded Vertex AI model
    """
    # Initialize Vertex AI SDK
    aiplatform.init(project=project_id, location=location)

    # Determine parent directory path in GCS
    if gcs_model_path.endswith(".pt"):
        # If path points to specific file, use its directory
        model_dir = os.path.dirname(gcs_model_path)

    # Create a model artifact in Vertex AI
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model_dir,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1.13:latest",  # Required by vertex Ai. Not going to use it
        description=description,
        labels=labels,
    )

    print(f"Model uploaded to Vertex AI Model Registry: {model.resource_name}")
    print(f"Model ID: {model.name}")
    return model


def load_model_from_vertex(
    model_name: str,
    project_id="swift-habitat-447619-k8",
    location="europe-west1",
):
    """
    Load a model from Vertex AI Model Registry.
    Args:
        model_name (str): Name of the model in Vertex AI Model Registry
        project_id (str): GCP project ID
        location (str): GCP region
    Returns:
        google.cloud.aiplatform.Model: The loaded Vertex AI model
    """
    # Initialize Vertex AI SDK
    aiplatform.init(project=project_id, location=location)

    # List models with the given display name
    models = aiplatform.Model.list(
        filter=f'display_name="{model_name}"', project=project_id, location=location
    )

    # Check if any models were found
    if not models:
        raise ValueError(f"No model found with name '{model_name}'")

    # Get the most recently created model if multiple have the same name
    model = models[0]
    if len(models) > 1:
        print(
            f"Warning: Multiple models found with name '{model_name}'. Using the most recent one."
        )
        # Sort by create_time (newest first)
        models.sort(key=lambda m: m.create_time, reverse=True)
        model = models[0]

    print(f"Found model: {model.display_name} (ID: {model.name})")

    model_uri: str = model.uri

    model_gcs_path = os.path.join(model_uri, "model.pt")

    # TODO: complete
    model = load_model_from_bucket()

    return model


if __name__ == "__main__":
    # upload_to_bucket("fineweb-datasets", "test.txt", "test.txt")
    # upload_directory_to_bucket(
    #     "fineweb-datasets",
    #     "data/tiny-dataset-processed.parquet",
    #     "tiny-dataset-processed",
    # )
    model = load_model_from_bucket("dummy-model")
    # print(model.uri)
    print(model(torch.tensor([5.0])))
