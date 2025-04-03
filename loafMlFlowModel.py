import os
import sys

import mlflow.pytorch


def load_model_from_mlflow():
    # Set MLflow server URL directly instead of using the database connection
    os.environ["MLFLOW_TRACKING_URI"] = "http://35.233.121.19:5000"

    # Model to load
    model_uri = "models:/FinewebEduClassifier/2"

    print(f"Attempting to load model from {model_uri}")
    print(f"Using MLflow tracking URI: {os.environ.get('MLFLOW_TRACKING_URI')}")

    try:
        # Load the model
        model = mlflow.pytorch.load_model(model_uri)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("Starting model loading script...")
    model = load_model_from_mlflow()

    if model is not None:
        print("Model info:", type(model))
        print("Script completed successfully")
    else:
        print("Script failed to load the model")
        sys.exit(1)
