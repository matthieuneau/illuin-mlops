import json
import os

import dotenv
import ray
import torch
from ray import serve
from starlette.requests import Request
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from gcpUtils import load_model_from_vertex

dotenv.load_dotenv()

runtime_env = {"pip": ["transformers", "torch", "pandas", "google-cloud-storage"]}

# Initialize Ray with runtime environment
ray.init(address="auto", runtime_env=runtime_env)


@serve.deployment(num_replicas=4, ray_actor_options={"num_cpus": 1, "num_gpus": 0})
class EduClassifierModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/fineweb-edu-classifier"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "HuggingFaceTB/fineweb-edu-classifier"
        )

    def run_inference(self, input_text) -> dict:
        input_tokens = self.tokenizer(
            input_text, return_tensors="pt", padding="longest", truncation=True
        )
        outputs = self.model(**input_tokens)
        logits = outputs.logits.squeeze(-1).float().detach().numpy()
        score = logits.item()
        result = {
            "text": input_text,
            "score": score,
            "int_score": int(round(max(0, min(score, 5)))),
        }
        return result

    async def __call__(self, http_request: Request) -> dict:
        # Parse the input from the request
        input_text = await http_request.json()

        # Run inference
        result = self.run_inference(input_text)

        # Return the result
        return result


translator_app = EduClassifierModel.bind()

# Connect to Ray cluster
# ray.init(address=f"ray://{os.getenv('RAY_ADDRESS')}:{os.getenv('RAY_SERVE_PORT')}")
# ray.init(address="auto")

# Start Ray Serve in detached mode
serve.start(detached=True, http_options={"host": "0.0.0.0"})
serve.run(translator_app, name="translator_app")
