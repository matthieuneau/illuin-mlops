import logging

from google.cloud import aiplatform

from gcpUtils import load_model_from_vertex

# List models with the given display name
model = load_model_from_vertex(
    "edu-classifier-v1", project_id="cs-3a-2024-fineweb-mlops", location="europe-west1"
)
print(model)
