import ray
from fastapi import FastAPI
from ray import serve
import torch

ray.init(
    address="ray://34.145.33.35:10001"
)  # TODO: check it is the correct head node ip
serve.start(http_options={"host": "0.0.0.0", "port": 8000})

app = FastAPI()

# Load the model once during app startup
model = torch.jit.load("models/classifier.pt")
model.eval()  # Set the model to evaluation mode


@serve.deployment
@serve.ingress(app)
class MyFastAPIDeployment:
    @app.get("/")
    def root(self):
        return "Hello, world!"

    @app.post("/predict_locally")
    async def predict(self, input_data: list[float]):
        input = torch.Tensor(input_data, dtype=torch.float32)
        with torch.no_grad():
            output = model(input)
        return {"output": output.tolist()}


serve.run(MyFastAPIDeployment.bind())
print(serve.status())
