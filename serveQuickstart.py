import os

import dotenv
import ray
from ray import serve
from starlette.requests import Request
from transformers import pipeline

dotenv.load_dotenv()


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 1, "num_gpus": 0})
class Translator:
    def __init__(self):
        # Load model
        self.model = pipeline("translation_en_to_fr", model="t5-small")

    def translate(self, text: str) -> str:
        # Run inference
        model_output = self.model(text)

        # Post-process output to return only the translation text
        translation = model_output[0]["translation_text"]

        return translation

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        return self.translate(english_text)


translator_app = Translator.bind()

# Connect to Ray cluster
# ray.init(address=f"ray://{os.getenv('RAY_ADDRESS')}:{os.getenv('RAY_SERVE_PORT')}")
ray.init(address="auto")

# Start Ray Serve in detached mode
serve.start(detached=True)
serve.run(translator_app, name="translator_app")
