import ray
import requests
from fastapi import FastAPI
from ray import serve

app = FastAPI()


@serve.deployment
@serve.ingress(app)
class MyFastAPIDeployment:
    @app.get("/")
    def root(self):
        return "Hello, world!"


serve.run(MyFastAPIDeployment.bind(), route_prefix="/hello")
resp = requests.get("http://localhost:8000/hello")
assert resp.json() == "Hello, world!"
