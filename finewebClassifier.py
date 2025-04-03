import os

import fsspec
import mlflow.pytorch
import ray
import torch
from dotenv import load_dotenv
from ray import serve
from starlette.requests import Request
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# dotenv.load_dotenv()
# os.environ["GOOGLE_CLOUD_PROJECT"] = "swift-habitat-447619-k8"
# os.environ["GCP_REGION"] = "europe-west1"
# os.environ["MODEL_ID"] = "4496012997138841600"


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 1, "num_gpus": 0})
class FinewebClassifier:
    def __init__(self):
        os.environ["MLFLOW_TRACKING_URI"] = (
            "postgresql://postgres:illuin1234@35.233.121.19:5432/ml_flow_db"
        )
        os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = "gs://mlflow-artifacts-illuin"

        # model_name = "FinewebEduClassifier"
        # model_stage = os.getenv(
        #     "MODEL_STAGE", "Production"
        # )  # Default to Production if not specified

        model_uri = "models:/FinewebEduClassifier/2"

        try:
            # Load the PyTorch model from MLflow
            self.model = mlflow.pytorch.load_model(model_uri)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from MLflow: {str(e)}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/fineweb-edu-classifier"
        )

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(
            text, return_tensors="pt", padding="longest", truncation=True
        )
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(-1).float().detach().numpy()
        score = logits.item()

        return score

    async def __call__(self, http_request: Request) -> str:
        text: str = await http_request.json()
        score = self.predict(text)
        result = {
            "text": text,
            "score": score,
            "int_score": int(round(max(0, min(score, 5)))),
        }
        return result


fineweb_classifier_app = FinewebClassifier.bind()

# Connect to Ray cluster
ray.init(address="auto")

# Start Ray Serve in detached mode
serve.start(detached=True, http_options={"host": "0.0.0.0"})
serve.run(fineweb_classifier_app, name="fineweb_classifier_app")
