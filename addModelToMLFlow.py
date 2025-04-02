import os

import mlflow
import mlflow.pytorch
from transformers import AutoModelForSequenceClassification

# mlflow.set_tracking_uri("http://34.38.116.114:5000")  # Set your MLflow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = (
    "postgresql://postgres:illuin1234@35.233.121.19:5432/ml_flow_db"
)
os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = "gs://mlflow-artifacts"


# Load the model directly from the Hugging Face Hub
model = AutoModelForSequenceClassification.from_pretrained(
    "HuggingFaceFW/fineweb-edu-classifier"
)

# Start an MLflow run and log the model
with mlflow.start_run() as run:
    mlflow.pytorch.log_model(model, artifact_path="model")
    run_id = run.info.run_id
    print("Run ID:", run_id)

# Construct the model URI from the run and register the model
model_uri = f"runs:/{run_id}/model"
registered_model = mlflow.register_model(model_uri, "FinewebEduClassifier")
print("Registered model:", registered_model.name, registered_model.version)
