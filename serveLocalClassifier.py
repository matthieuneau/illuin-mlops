import json
import time

import dotenv
import ray
from ray import serve

# import debugpy
from ray.serve.handle import DeploymentHandle
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from gcpUtils import load_model_from_vertex

# from gcpUtils import load_model_from_vertex

dotenv.load_dotenv()

runtime_env = {
    "pip": [
        "transformers",
        "torch",
        "pandas",
        "google-cloud-storage",
        "google-cloud-aiplatform",
        "prometheus_client",
    ],
    # "env_vars": {"RAY_DEBUG": "1", "RAY_DEBUG_POST_MORTEM": "1"},
}

# Initialize Ray with runtime environment
ray.init(
    address="auto",
    runtime_env=runtime_env,
)


# We'll use a separate deployment for metrics collection
@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
class MetricsCollector:
    def __init__(self):
        # Import prometheus_client here to avoid serialization issues
        from prometheus_client import Counter, Histogram, Summary

        # Initialize metrics
        self.request_counter = Counter(
            "edu_classifier_requests_total", "Total number of classification requests"
        )
        self.model_score_summary = Summary(
            "edu_classifier_score_summary",
            "Distribution of educational content classifier scores",
        )
        self.model_score_bucket = Counter(
            "edu_classifier_score_bucket",
            "Count of scores by integer value",
            ["score_bucket"],
        )
        self.inference_latency = Histogram(
            "edu_classifier_inference_seconds", "Time spent processing inference"
        )
        print("Metrics collector initialized")

    async def __call__(self, request: Request) -> Response:
        import prometheus_client

        if request.url.path == "/metrics":
            # Serve the metrics endpoint
            metrics_data = prometheus_client.generate_latest()
            return Response(
                content=metrics_data,
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

        # For other paths, return metrics info
        return JSONResponse(
            {
                "message": "This is the metrics collector endpoint. Use /metrics to scrape Prometheus metrics."
            }
        )

    def record_request(self):
        """Increment the request counter"""
        self.request_counter.inc()

    def record_score(self, int_score):
        """Record a model score"""
        self.model_score_bucket.labels(score_bucket=str(int_score)).inc()

    def record_latency(self, seconds):
        """Record inference latency"""
        self.inference_latency.observe(seconds)


# @serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0})
class EduClassifierModel:
    def __init__(
        self,
        model_name: str = "edu-classifier-v1",
        project_id: str = "cs-3a-2024-fineweb-mlops",
        location: str = "europe-west1",
    ):
        """
        Args:
            model_name (str): The name of the model as in vertex ai model registry
            project_id (str): The GCP project ID
            location(str): The GCP region where the model is hosted
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/fineweb-edu-classifier"
        )
        self.model = load_model_from_vertex(
            model_name=model_name,
            project_id=project_id,
            location=location,
        )

        # We'll use handle to communicate with the metrics collector
        self.metrics_collector = serve.get_app_handle("metrics_collector")

    def run_inference(self, input_text) -> dict:
        import torch

        # Track inference time
        start_time = time.time()

        input_tokens = self.tokenizer(
            input_text, return_tensors="pt", padding="longest", truncation=True
        )
        del input_tokens[
            "token_type_ids"
        ]  # Remove token_type_ids if present. Invalid arg for jit compiled models
        outputs = self.model(**input_tokens)
        # logits = outputs.logits.squeeze(-1).float().detach().numpy()
        int_score = torch.argmax(outputs).item()

        # Record latency
        latency = time.time() - start_time

        # Record metrics asynchronously - handle DeploymentResponse correctly
        self.metrics_collector.record_request.remote()
        self.metrics_collector.record_score.remote(int_score)
        self.metrics_collector.record_latency.remote(latency)

        return {"score": int_score}

    async def __call__(self, http_request: Request) -> Response:
        import prometheus_client

        # IMPORTANT: Check the path BEFORE trying to parse JSON
        if http_request.url.path == "/metrics":
            try:
                # Generate metrics
                metrics_data = prometheus_client.generate_latest()

                # Return a simple text response
                return Response(
                    content=metrics_data,
                    media_type="text/plain; version=0.0.4; charset=utf-8",
                )
            except Exception as metrics_error:
                print(f"Error generating metrics: {str(metrics_error)}")
                return Response(
                    content=f"Metrics error: {str(metrics_error)}", status_code=500
                )

        try:
            # Increment request counter for regular requests
            self.metrics_collector.record_request.remote()

            # Only try to parse JSON for non-metrics requests
            input_text = await http_request.json()

            # Run inference
            result = self.run_inference(input_text)

            # Return a JSON response
            return Response(content=json.dumps(result), media_type="application/json")

        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=500,
                media_type="application/json",
            )


# Entry point of our application
@serve.deployment(num_replicas=1)
class Ingress:
    def __init__(
        self, classifier_french: DeploymentHandle, classifier_english: DeploymentHandle
    ):
        self.classifier_french_handle = classifier_french
        self.classifier_english_handle = classifier_english
        print("Ingress initialized with classifier handles")

    async def __call__(self, request: Request) -> Response:
        from transformers import pipeline

        # Basic health check
        if request.url.path == "/health":
            return JSONResponse({"status": "healthy"})

        # Process the request locally instead of forwarding
        try:
            # Try to parse JSON for normal requests
            try:
                request_json = await request.json()
                if isinstance(request_json, dict) and "text" in request_json:
                    input_text = request_json["text"]
                else:
                    input_text = json.dumps(request_json)
            except:
                try:
                    input_text = await request.body()
                    input_text = input_text.decode("utf-8")
                except:
                    input_text = "Failed to parse request body"

            lang_detector = pipeline(
                "text-classification",
                model="papluca/xlm-roberta-base-language-detection",
            )
            text_lang = lang_detector(input_text)

            # Forward to the appropriate classifier based on language
            if text_lang[0]["label"] == "fr":
                result = await self.classifier_french_handle.run_inference.remote(
                    input_text
                )
            elif text_lang[0]["label"] == "en":
                result = await self.classifier_english_handle.run_inference.remote(
                    input_text
                )
            else:
                return JSONResponse(
                    content={"error": f"Unsupported language: {text_lang[0]['label']}"},
                    status_code=400,
                )

            return JSONResponse(content=result)
        except Exception as e:
            import traceback

            error_trace = traceback.format_exc()
            print(f"Error in Ingress.__call__: {str(e)}\n{error_trace}")
            return JSONResponse(
                content={"error": f"Ingress error: {str(e)}"},
                status_code=500,
            )


@serve.deployment(name="classifier_french", num_replicas=1)
class FrenchClassifier(EduClassifierModel):
    def __init__(self):
        super().__init__(
            model_name="edu-classifier-v1",
            location="europe-west1",
            project_id="cs-3a-2024-fineweb-mlops",
        )


@serve.deployment(name="classifier_english", num_replicas=1)
class EnglishClassifier(EduClassifierModel):
    def __init__(self):
        super().__init__(
            model_name="edu-classifier-v1",
            location="europe-west1",
            project_id="cs-3a-2024-fineweb-mlops",
        )


def launch_application():
    metrics_collector = MetricsCollector.bind()
    time.sleep(2)  # Wait for the metrics collector to initialize
    serve.run(metrics_collector, name="metrics_collector", route_prefix="/metrics")
    print("Metrics collector deployed")

    classifier_deployment_french = FrenchClassifier.bind()
    classifier_deployment_english = EnglishClassifier.bind()

    app = Ingress.bind(classifier_deployment_french, classifier_deployment_english)

    # Start Ray Serve in detached mode
    serve.start(detached=True, http_options={"host": "0.0.0.0"})
    app = serve.run(app, name="app", route_prefix="/")


if __name__ == "__main__":
    import requests

    # launch_application()

    url = "http://localhost:8000/"
    headers = {"Content-Type": "text/plain"}
    data = "The Pilgrims, also known as the Pilgrim Fathers, were the English settlers who travel around the world until infinity"

    response = requests.post(url, headers=headers, data=data)

    print(response)
