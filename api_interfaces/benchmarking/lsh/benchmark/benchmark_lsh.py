import pandas as pd

import os
import time
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm
from wasabi import Printer

msg = Printer()

data_path = Path("/data")
image_ids = np.load(data_path/"ids.npy")


def get_response(n_classifiers, n_clusters, image_id=None):
    image_id = image_id or np.random.choice(image_ids)
    response = requests.get(
        url="http://0.0.0.0/similar-images/approximate/",
        params={
            "n_classifiers": n_classifiers,
            "n_clusters": n_clusters,
            "n": 6,
            "image_id": image_id
        }
    ).json()
    return response


def benchmark(n_classifiers, n_clusters, sample_size=250):
    loop = tqdm(
        [],
        total=sample_size,
        ncols=100,
        desc=f"Running test for index {n_classifiers}-{n_clusters}"
    )
    request_times, n_failures = [], 0
    while len(request_times) < sample_size:
        try:
            image_id = np.random.choice(image_ids)
            start_time = time.time()
            _ = get_response(n_classifiers, n_clusters, image_id)
            request_times.append(time.time() - start_time)
            loop.update()
        except:
            n_failures += 1

    time_per_request = sum(request_times) / len(request_times)
    rounded_time = round(time_per_request, 4) * 1000
    msg.warn(f"{n_failures} failures")
    msg.good(f"Average request time: {rounded_time}")
    return time_per_request


if __name__ == "__main__":
    n_classifiers_options = [32, 64, 128, 256, 512]
    n_clusters_options = [8, 16, 32, 64, 128, 256]
    results = pd.DataFrame({
        n_classifiers: {
            n_clusters: None
            for n_clusters in n_clusters_options}
        for n_classifiers in n_classifiers_options
    })
    for n_classifiers in n_classifiers_options:
        for n_clusters in n_clusters_options:
            results[n_classifiers][n_clusters] = benchmark(
                n_classifiers, n_clusters
            )
    print(results)
