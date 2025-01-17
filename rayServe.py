import ray
import io
from fastapi import FastAPI
from google.cloud import storage
from ray import serve
import torch
import os

# TODO: Remove runtime_env when deploying to cloud
ray.init(address="auto")

# TODO: check if needed to specify address
serve.start(http_options={"host": "0.0.0.0", "port": 8000})

app = FastAPI()


@serve.deployment
@serve.ingress(app)
class MyFastAPIDeployment:
    def __init__(self) -> None:
        model_path = os.path.expanduser("/home/matthieuneau/models/classifier.pt")
        self.model = torch.jit.load(model_path)
        self.model.eval()  # Set the model to evaluation mode

    @app.get("/")
    def root(self):
        return "Hello, world!"

    @app.post("/predict_locally")
    async def predict(self, input_data: list[float]):
        """This function calls a model that is already available on the cluster"""
        input = torch.tensor([input_data], dtype=torch.float32)
        with torch.no_grad():
            output = self.model(input)
        return {"output": output.tolist()}

    @app.post("/predict_from_s3")
    async def predict_s3(self, input_data: list[float]):
        """This function downloads the model from gcp bucket every time there is a request"""
        bucket_name = "fineweb-classifiers"
        model_blob_name = "classifier.pt"

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(model_blob_name)

        model_bytes = blob.download_as_bytes()

        model_gcp_bucket = torch.jit.load(io.BytesIO(model_bytes))
        model_gcp_bucket.eval()

        input = torch.tensor([input_data], dtype=torch.float32)

        with torch.no_grad():
            output = model_gcp_bucket(input)
        return {"output": output.tolist()}


serve.run(MyFastAPIDeployment.bind())
print(serve.status())
