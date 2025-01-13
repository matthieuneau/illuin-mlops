from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from predict import predict

app = FastAPI()


MODEL_BUCKET_NAME = "fineweb-classifiers"
MODEL_FILE_NAME = "classifier.pt"
LOCAL_MODEL_PATH = "model.pkl"


@app.get("/")
def hello_world():
    return {"message": "hello_world"}


class PredictRequest(BaseModel):
    model_bucket: str
    model_path: str
    data_bucket: str
    data_path: str


@app.post("/predict")
async def predict_endpoint(request: PredictRequest):
    try:
        predictions = predict(
            request.model_bucket,
            request.model_path,
            request.data_bucket,
            request.data_path,
        )
        return {"preditions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
