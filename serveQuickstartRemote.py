import os

import dotenv
import ray
import torch
from ray import serve
from starlette.requests import Request
from transformers import pipeline

from gcpUtils import load_model_from_vertex

dotenv.load_dotenv()

# Initialize Ray with runtime environment
ray.init(address="auto")


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 1, "num_gpus": 0})
class Model:
    def __init__(self):
        # Load model
        self.model = load_model_from_vertex(model_name="dummy_model")

    def run_inference(self, input) -> torch.Tensor:
        input = torch.tensor(input)
        output = self.model(input)
        return output

    async def __call__(self, http_request: Request) -> str:
        # Parse the input from the request
        input_data = await http_request.json()

        # Run inference
        result = self.run_inference(input_data)

        # Convert tensor output to Python native type for JSON response
        if isinstance(result, torch.Tensor):
            result = result.item() if result.numel() == 1 else result.tolist()

        # Return the result
        return {"result": result}


app = Model.bind()

# Start Ray Serve in detached mode
serve.start(detached=True, http_options={"host": "0.0.0.0"})
serve.run(app, name="app")
