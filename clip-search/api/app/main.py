import torch
import clip
from pydantic import BaseModel
from typing import Literal, Annotated, Optional
import os

from fastapi import FastAPI, Query
from .src.elasticsearch import get_prototype_elastic_client
from .src.log import get_logger

log = get_logger()

log.info("Establishing connection to Elasticsearch")
es = get_prototype_elastic_client()
index_name = f"images-clip-{os.environ['INDEX_DATE']}"

log.info("Loading CLIP model")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = os.environ["MODEL_NAME"]
model, preprocessor = clip.load(
    model_name, download_root="/data/models", device=device
)


app = FastAPI()

default_includes = ["source_id", "image_id", "thumbnail_url", "title"]
allowed_includes = Literal["embedding"]


class Document(BaseModel):
    score: Optional[float]
    source_id: str
    image_id: str
    title: str
    thumbnail_url: str
    embedding: Optional[list[float]]


class Response(BaseModel):
    results: dict[str, Document]
    total: int
    took: int


@app.get("/")
def health_check() -> dict:
    return {"status": "healthy"}


@app.get("/images")
def get(
    query: Optional[str] = None,
    n: Optional[int] = 10,
    includes: Annotated[list[allowed_includes] | None, Query()] = [],
) -> Response:
    """
    Get images, optionally filtered by a query string
    """
    if not query:
        return _get_all(includes=includes)
    else:
        return _get_nearest(query=query, n=n, includes=includes)


@app.get("/images/{id}")
def get_by_id(
    id: str, includes: Annotated[list[allowed_includes] | None, Query()] = []
) -> Response:
    """Get images, filtered by work id"""
    response = es.search(
        index=index_name,
        query={"match": {"image_id": id}},
        _source=default_includes + includes,
    )
    return {
        "results": {
            hit["_id"]: hit["_source"] for hit in response["hits"]["hits"]
        },
        "total": response["hits"]["total"]["value"],
        "took": response["took"],
    }


@app.get("/embed")
def embed(query: str) -> dict:
    """Get an embedding for a supplied string"""
    embedding = _get_embedding(query)
    return {"embedding": embedding, "model": model_name}


def _get_embedding(search_terms: str) -> list[float]:
    with torch.no_grad():
        tokens = clip.tokenize([search_terms]).to(device)
        text_embedding = model.encode_text(tokens).squeeze(0)

    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    return text_embedding.tolist()


def _get_nearest(query, n, includes) -> Response:
    """Get nearest images in an index for a supplied string"""
    embedding = _get_embedding(query)
    knn_query = {
        "field": "embedding",
        "query_vector": embedding,
        "k": 1000,
        "num_candidates": 1000,
    }

    response = es.search(
        index=index_name,
        knn=knn_query,
        _source=default_includes + includes,
        size=n,
    )
    return {
        "results": {
            hit["_id"]: {**hit["_source"], "score": hit["_score"]}
            for hit in response["hits"]["hits"]
        },
        "total": response["hits"]["total"]["value"],
        "took": response["took"],
    }


def _get_all(includes) -> Response:
    """Get all images"""
    response = es.search(
        index=index_name,
        collapse={"field": "id"},
        _source=default_includes + includes,
        query={"match_all": {}},
    )
    return {
        "results": {
            hit["_id"]: hit["_source"] for hit in response["hits"]["hits"]
        },
        "total": response["hits"]["total"]["value"],
        "took": response["took"],
    }
