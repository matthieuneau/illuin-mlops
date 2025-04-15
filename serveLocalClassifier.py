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

# from gcpUtils import load_model_from_vertex

dotenv.load_dotenv()

runtime_env = {
    "pip": ["transformers", "torch", "pandas", "google-cloud-storage"],
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

    def record_score(self, score, int_score):
        """Record a model score"""
        self.model_score_summary.observe(score)
        self.model_score_bucket.labels(score_bucket=str(int_score)).inc()

    def record_latency(self, seconds):
        """Record inference latency"""
        self.inference_latency.observe(seconds)


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0})
class EduClassifierModel:
    def __init__(self, name="EduClassifierModel1"):
        # print("waiting for debugger to attach...")
        # debugpy.wait_for_client()
        # print("debugger attached")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/fineweb-edu-classifier"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "HuggingFaceTB/fineweb-edu-classifier"
        )
        # We'll use handle to communicate with the metrics collector
        self.metrics_collector = serve.get_app_handle("metrics_collector")

    def run_inference(self, input_text) -> dict:
        # Track inference time
        start_time = time.time()

        input_tokens = self.tokenizer(
            input_text, return_tensors="pt", padding="longest", truncation=True
        )
        outputs = self.model(**input_tokens)
        logits = outputs.logits.squeeze(-1).float().detach().numpy()
        score = logits.item()
        int_score = int(round(max(0, min(score, 5))))

        # Record latency
        latency = time.time() - start_time

        # Record metrics asynchronously - handle DeploymentResponse correctly
        self.metrics_collector.record_request.remote()
        self.metrics_collector.record_score.remote(score, int_score)
        self.metrics_collector.record_latency.remote(latency)

        result = {
            "text": input_text,
            "score": score,
            "int_score": int_score,
        }
        return result

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
        self, classifier_app1: DeploymentHandle, classifier_app2: DeploymentHandle
    ):
        self.classifier1_handle = classifier_app1
        self.classifier2_handle = classifier_app2
        print("Ingress initialized with classifier handles")

    async def __call__(self, request: Request) -> Response:
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

            # Forward with text parameter directly instead of the whole request
            result = await self.classifier1_handle.run_inference.remote(input_text)
            return JSONResponse(content=result)
        except Exception as e:
            import traceback

            error_trace = traceback.format_exc()
            print(f"Error in Ingress.__call__: {str(e)}\n{error_trace}")
            return JSONResponse(
                content={"error": f"Ingress error: {str(e)}"},
                status_code=500,
            )


metrics_collector = MetricsCollector.bind()
classifier_app1 = EduClassifierModel.bind(name="EduClassifierModel1")
classifier_app2 = EduClassifierModel.bind(name="EduClassifierModel2")

app = Ingress.bind(classifier_app1, classifier_app2)

# Connect to Ray cluster
# ray.init(address=f"ray://{os.getenv('RAY_ADDRESS')}:{os.getenv('RAY_SERVE_PORT')}")
# ray.init(address="auto")

# Start Ray Serve in detached mode
serve.start(detached=True, http_options={"host": "0.0.0.0"})
serve.run(metrics_collector, name="metrics_collector", route_prefix="/metrics")

while True:
    status = serve.status()
    if "metrics_collector" not in status.applications:
        print(
            "Waiting for metrics collector to start before deploying classifiers, retrying in 1 second..."
        )
        time.sleep(1)
    break

app = serve.run(app, name="app", route_prefix="/")
