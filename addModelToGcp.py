import datetime
import os

from dotenv import load_dotenv
from google.cloud import aiplatform, storage
from huggingface_hub import snapshot_download

load_dotenv()

# Download the model from Hugging Face Hub
# local_dir = snapshot_download("HuggingFaceFW/fineweb-edu-classifier")

project = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
bucket_name = "fineweb-classifiers"  # Ensure this bucket already exists

local_model_dir = os.path.expanduser(
    "~/.cache/huggingface/hub/models--HuggingFaceFW--fineweb-edu-classifier/snapshots/284663cbb2dabf9bda30d8f8cc49601251ee1631"
)
# The folder inside the bucket where the model will be uploaded.
gcs_model_dir = (
    f"fineweb-edu-classifier/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}/"
)

storage_client = storage.Client(project=project)
bucket = storage_client.bucket(bucket_name)

for root, dirs, files in os.walk(local_model_dir):
    for file in files:
        local_file_path = os.path.join(root, file)
        # Determine the relative path inside the model directory.
        relative_path = os.path.relpath(local_file_path, local_model_dir)
        # Create the destination blob path.
        blob_path = os.path.join(gcs_model_dir, relative_path)
        blob = bucket.blob(blob_path)
        print(f"Uploading {local_file_path} to gs://{bucket_name}/{blob_path}")
        blob.upload_from_filename(local_file_path)


aiplatform.init(project=project, location=location)

# Use the GCS path for the uploaded model directory.
gcs_artifact_uri = f"gs://{bucket_name}/{gcs_model_dir}"

model = aiplatform.Model.upload(
    display_name="fineweb-edu-classifier",
    artifact_uri=gcs_artifact_uri,
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tensorflow-serving:latest",  # Dummy required to upload. But we'll use ray cluster
)

print("Model uploaded, Resource name:", model.resource_name)
