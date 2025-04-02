import os

import dotenv
import fsspec
import ray
import torch
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
        model_uri = f"projects/{os.getenv('GOOGLE_CLOUD_PROJECT')}/locations/{os.getenv('GCP_REGION')}/models/{os.getenv('MODEL_ID')}"
        if not model_uri:
            raise ValueError("MODEL_REGISTRY_URI environment variable is not set.")

        # Open the model file directly from GCS using fsspec
        with fsspec.open(model_uri, "rb") as f:
            self.model = torch.load(f)

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
# ray.init(address=f"ray://{os.getenv('RAY_ADDRESS')}:{os.getenv('RAY_SERVE_PORT')}")

# Start Ray Serve in detached mode
# serve.start(detached=True, http_options={"host": "0.0.0.0"})
# serve.run(fineweb_classifier_app, name="fineweb_classifier_app")
