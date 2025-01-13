from fastapi import FastAPI, HTTPException
from google.cloud import storage
import joblib
import os

app = FastAPI()


MODEL_BUCKET_NAME = "fineweb-classifiers"
MODEL_FILE_NAME = "classifier.pt"
LOCAL_MODEL_PATH = "model.pkl"


@app.get("/")
def hello_world():
    return {"message": "hello_world"}


def load_model_from_gcp(bucket_name: str, model_file_name: str, local_model_path: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_file_name)

    blob.download_to_filename(local_model_path)
    print(f"Model downloaded to {local_model_path}")

    return joblib.load(local_model_path)
