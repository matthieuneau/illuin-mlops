import ray
from fastapi import FastAPI
from ray import serve
import torch

# TODO: Remove runtime_env when deploying to cloud
ray.init(address="auto")

# TODO: check if needed to specify address
serve.start(http_options={"host": "0.0.0.0", "port": 8000})

app = FastAPI()


@serve.deployment
@serve.ingress(app)
class MyFastAPIDeployment:
    def __init__(self) -> None:
        self.model = torch.jit.load("~/models/classifier.pt")
        self.model.eval()  # Set the model to evaluation mode

    @app.get("/")
    def root(self):
        return "Hello, world!"

    @app.post("/predict_locally")
    async def predict(self, input_data: list[float]):
        input = torch.tensor(input_data, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(input)
        return {"output": output.tolist()}


serve.run(MyFastAPIDeployment.bind())
print(serve.status())
