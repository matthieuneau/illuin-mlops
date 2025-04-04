import os

import ray
from dotenv import load_dotenv
from google.cloud import aiplatform
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
        model_name = "fineweb-edu-classifier"
        # TODO: figure out how to enforce version
        version = "1"
        model = aiplatform.Model.list(
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ["GCP_REGION"],
            filter=f'display_name="{model_name}" ',
        )[0]

        # Download model artifacts to a local directory
        local_model_path = model.download()

        # Load the tokenizer and model from the downloaded path
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            local_model_path
        )
        self.model.eval()

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(
            text, return_tensors="pt", padding="longest", truncation=True
        )
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(-1).float().detach().numpy()
        score = logits.item()

        return score

    async def __call__(self, http_request: Request) -> dict:
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
