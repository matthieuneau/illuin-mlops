from fastapi import FastAPI
import requests
import torch

app = FastAPI()

model = torch.jit.load("models/classifier.pt")
model.eval()  # Set the model to evaluation mode


@app.get("/")
def root():
    return "Hello, world!"


@app.post("/predict_locally")
async def predict(input_data: list[float]):
    input = torch.tensor([input_data], dtype=torch.float32)
    with torch.no_grad():
        output = model(input)
    return {"output": output.tolist()}


if __name__ == "__main__":
    test_input = torch.rand(384).tolist()
    with open("input.txt", "w") as f:
        f.write(str(test_input))
        f.close()
    response = requests.post("http://localhost:8000/predict_locally", json=test_input)
    print(response.json())
