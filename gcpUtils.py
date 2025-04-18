import os
import tempfile

import google.auth
import pandas as pd
import torch
from dotenv import load_dotenv
from google.cloud import aiplatform, storage

load_dotenv()


def upload_to_bucket(
    bucket_name: str, source_file_path: str, destination_blob_name: str
):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_path)


def load_model_from_bucket(
    bucket_name: str, folder_name: str = "model_init", file_name: str = "model.pt"
):
    """
    Load a TorchScript model from a GCP bucket.

    Args:
        bucket_name (str): Name of the GCP bucket.
        folder_name (str): Name of the folder within the bucket. Default is "model_init".
        file_name (str): Name of the model file. Default is "model.pt".

    Returns:
        torch.jit.ScriptModule: The loaded TorchScript model.
    """
    import os
    import tempfile

    import torch
    from google.cloud import storage

    client = storage.Client(project="cs-3a-2024-fineweb-mlops")
    bucket = client.bucket(bucket_name)

    # Construct the blob path using folder_name and file_name
    blob_path = f"{folder_name}/{file_name}"
    blob = bucket.blob(blob_path)

    # Create a named temporary file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Download the model to the temporary file
        blob.download_to_filename(temp_path)

        # Load the model
        model = torch.jit.load(temp_path)
        return model
    finally:
        # Clean up the temporary file
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
    project_id="cs-3a-2024-fineweb-mlops",
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
    project_id="cs-3a-2024-fineweb-mlops",
    location="us-central1",
):
    """
    Load a model from Vertex AI Model Registry. It determines the bucket in which the model is stored and loads it
    using the load_model_from_bucket function.
    Args:
        model_name (str): Name of the model in Vertex AI Model Registry
        project_id (str): GCP project ID
        location (str): GCP region
    Returns:
        google.cloud.aiplatform.Model: The loaded Vertex AI model
    """
    credentials, project = google.auth.default()

    # Initialize Vertex AI SDK
    aiplatform.init(project=project_id, location=location, credentials=credentials)

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
    bucket_name = model_uri.split("/")[2]
    folder_name = model_uri.split("/")[3]

    model = load_model_from_bucket(
        bucket_name,
        folder_name=folder_name,
    )

    return model, model_uri


def create_big_query_dataset(
    project_id: str,
    dataset_id: str,
    location: str = "europe-west1",
):
    """
    Create a BigQuery dataset.

    Args:
        project_id (str): GCP project ID
        dataset_id (str): Dataset ID to create

    Returns:
        None
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)

    # Define the dataset reference
    dataset_ref = f"{project_id}.{dataset_id}"

    # Create a Dataset object
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = location

    # Create the dataset in BigQuery
    client.create_dataset(dataset, exists_ok=True)


def create_big_query_table(
    project_id: str,
    dataset_id: str,
    table_id: str,
    schema: list,
):
    """
    Create a BigQuery table with the specified schema.

    Args:
        project_id (str): GCP project ID
        dataset_id (str): Dataset ID in BigQuery
        table_id (str): Table ID to create
        schema (list): List of dictionaries defining the schema

    Returns:
        None
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)

    # Define the table reference
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    # Create a Table object
    table = bigquery.Table(table_ref, schema=schema)

    # Create the table in BigQuery
    client.create_table(table)


if __name__ == "__main__":
    # upload_gcs_model_to_vertex(
    #     gcs_model_path="gs://fineweb_models/model_init/model_init_script.pt",
    #     display_name="edu-classifier-v1",
    #     project_id="cs-3a-2024-fineweb-mlops",
    #     location="us-central1",
    #     description="First edu classifier model that handles all the requests for now. We will have one for each language in the future",
    # )
    # upload_directory_to_bucket(
    #     "fineweb-datasets",
    #     "data/tiny-dataset-processed.parquet",
    #     "tiny-dataset-processed",
    # )
    # model = load_model_from_bucket("dummy-model")
    model = load_model_from_vertex(
        "edu-classifier-v1",
        location="europe-west1",
        project_id="cs-3a-2024-fineweb-mlops",
    )
    # print(model.uri.split("/")[2])
    # print(model(torch.tensor([5.0])))
