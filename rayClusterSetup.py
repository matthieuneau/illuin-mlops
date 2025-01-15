from collections import Counter
import socket
import time

import ray

ray.init()

print(
    f"This cluster consists of {len(ray.nodes())} nodes and {ray.cluster_resources()['CPU']} CPU"
)


@ray.remote
def f():
    time.sleep(0.001)
    # Return IP address.
    return socket.gethostbyname("localhost")


object_ids = [f.remote() for _ in range(5000)]
ip_addresses = ray.get(object_ids)

print("tasks executed")
print(Counter(ip_addresses))
