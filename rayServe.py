import ray
import requests
from fastapi import FastAPI
from ray import serve

ray.init(
    address="ray://34.145.33.35:10001"
)  # TODO: check it is the correct head node ip
serve.start(http_options={"host": "0.0.0.0", "port": 8000})

app = FastAPI()


@serve.deployment
@serve.ingress(app)
class MyFastAPIDeployment:
    @app.get("/")
    def root(self):
        return "Hello, world!"


serve.run(MyFastAPIDeployment.bind(), route_prefix="/hello")
print(serve.status())
