from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from elastic import ES

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

es = ES()
app = FastAPI()
data_path = Path("/data")
image_ids = np.load(data_path / "ids.npy")


def id_to_url(image_id):
    return f"https://iiif.wellcomecollection.org/image/{image_id}.jpg/full/760,/0/default.jpg"


@app.get("/similar-images/approximate")
def approximate(n_classifiers: int, n_clusters: int, image_id: str = None, n: int = 10):
    image_id = image_id or np.random.choice(image_ids)
    index_name = f"{n_classifiers}-{n_clusters}"
    similar_image_ids = es.lsh_query(index_name, image_id, n)
    response = {
        "query_id": image_id,
        "query_image_url": id_to_url(image_id),
        "similar_image_ids": similar_image_ids,
        "similar_image_urls": [
            id_to_url(image_id) for image_id in similar_image_ids
        ]
    }
    return response


@app.get("/similar-images/exact")
def exact(image_id: str = None, n: int = 10):
    image_id = image_id or np.random.choice(image_ids)
    similar_image_ids = es.exact_query(image_id, n)
    response = {
        "query_id": image_id,
        "query_image_url": id_to_url(image_id),
        "similar_image_ids": similar_image_ids,
        "similar_image_urls": [
            id_to_url(image_id) for image_id in similar_image_ids
        ]
    }
    return response


@app.post("/assessment",)
async def post_assessment(assessment: dict):
    es.index_document(
        index_name="assessment",
        body=assessment
    )
    return assessment


@app.get("/data_for_interface")
def data_for_interface():
    query_id = np.random.choice(image_ids)
    index_a = random_index_name()
    index_b = random_index_name()
    while index_b == index_a:
        index_b = random_index_name()
    similar_image_ids_a = es.lsh_query(index_a, query_id, 6)
    similar_image_ids_b = es.lsh_query(index_b, query_id, 6)

    return {
        "query_id": query_id,
        "index_a": index_a,
        "index_b": index_b,
        "similar_image_ids_a": similar_image_ids_a,
        "similar_image_ids_b": similar_image_ids_b,
    }


def random_index_name():
    n_classifiers = np.random.choice([32, 64, 128, 256, 512])
    n_clusters = np.random.choice([8, 16, 32, 64, 128, 256])
    return str(n_classifiers) + "-" + str(n_clusters)
