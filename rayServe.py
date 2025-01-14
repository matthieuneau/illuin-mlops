from typing import Dict

from fastapi import FastAPI
import requests
from ray import serve
import starlette
from starlette.requests import Request
import torch

app = FastAPI()


@serve.deployment
class Classifier:
    def __init__(self) -> None:
        self.model = torch.jit.load("models/classifier.pt").eval()

    async def __call__(self, starlette_request: Request) -> dict:
        payload_bytes = await starlette_request.body()
        with torch.no_grad():
            output_tensor = self.model(payload_bytes)
        return {
            "results": int(torch.argmax(output_tensor, dim=1))
        }  # TODO: check if its the right dim


@serve.deployment
@serve.ingress(app)
class FastAPIDeployment:
    @app.get("/hello")
    def say_hello(self, name: str) -> str:
        return f"hello {name}"


# 2: Deploy the application locally.
serve.run(FastAPIDeployment.bind(), route_prefix="/")

# 3: Query the application and print the result.
print(requests.get("http://localhost:8000/hello", params={"name": "Theo"}).json())
