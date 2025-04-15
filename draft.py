import ray
from ray import serve

# Connect to Ray
ray.init(address="auto")

# Get the deployment
deployment = serve.get_deployment_handle("translator_app", "EduClassifierModel")

# Get handles to the replicas
handles = deployment.get_handles()

# Kill each replica
for handle in handles:
    ray.kill(handle, no_restart=True)
    print(f"Killed actor: {handle}")
