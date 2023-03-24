from typing import Literal, Annotated, Optional
import os

from fastapi import FastAPI, Query
from .src.embed import TextEmbedder
from .src.elasticsearch import get_elastic_client

target_es = get_elastic_client()
model_name = os.environ["MODEL_NAME"]

model = TextEmbedder(model=model_name, cache_dir="/data/embeddings")

app = FastAPI()

default_includes = ["id", "type", "title", "text", "format"]
allowed_includes = Literal["embedding"]
allowed_prismic_formats = Literal[
    "articles",
    "webcomics",
    "events",
    "exhibitions",
    "books",
    "pages",
]
allowed_works_formats = Literal[
    "Ephemera", "Books", "Pictures", "Archives and Manuscripts", "Digital Images", "Videos"
]


@app.get("/works")
def works(
    query: Optional[str] = None,
    n: Optional[int] = 10,
    includes: Annotated[list[allowed_includes] | None, Query()] = [],
    formats: Annotated[list[allowed_works_formats] | None, Query()] = [],
):
    if not query:
        return _get_all(
            index=f"works-{model_name}", includes=includes, formats=formats
        )
    else:
        return _get_nearest(
            index=f"works-{model_name}",
            query=query,
            n=n,
            includes=includes,
            formats=formats,
        )


@app.get("/prismic")
def prismic(
    query: Optional[str] = None,
    n: Optional[int] = 10,
    includes: Annotated[list[allowed_includes] | None, Query()] = [],
    formats: Annotated[list[allowed_prismic_formats] | None, Query()] = [],
):
    if not query:
        return _get_all(
            index=f"prismic-{model_name}", includes=includes, formats=formats
        )
    else:
        return _get_nearest(
            index=f"prismic-{model_name}",
            query=query,
            n=n,
            includes=includes,
            formats=formats,
        )


def _get_nearest(index, query, n, includes, formats):
    """Get nearest embeddings in an index for a supplied string"""
    embedding = model.embed(query)
    response = target_es.search(
        index=index,
        knn={
            "field": "embedding",
            "query_vector": embedding,
            "k": 1000,
            "num_candidates": 1000,
        },
        collapse={"field": "id"},
        _source=default_includes + includes,
        size=n,
        query={"bool": {"filter": [{"terms": {"format": formats}}]}}
        if formats
        else {"match_all": {}},
    )
    return {
        "embeddings": {
            hit["_id"]: {**hit["_source"], "score": hit["_score"]}
            for hit in response["hits"]["hits"]
        },
        "total": response["hits"]["total"]["value"],
        "took": response["took"],
    }


def _get_all(index, includes, formats):
    """Get all embeddings in an index"""
    response = target_es.search(
        index=index,
        collapse={"field": "id"},
        _source=default_includes + includes,
        query={"bool": {"filter": [{"terms": {"format": formats}}]}}
        if formats
        else {"match_all": {}},
    )
    return {
        "embeddings": {
            hit["_id"]: hit["_source"] for hit in response["hits"]["hits"]
        },
        "total": response["hits"]["total"]["value"],
        "took": response["took"],
    }


@app.get("/works/{id}")
def get_work(
    id: str, includes: Annotated[list[allowed_includes] | None, Query()] = []
):
    """Get embeddings, filtered by work id"""
    response = target_es.search(
        index=f"works-{model_name}",
        body={
            "query": {"match": {"id": id}},
            "sort": [{"id": {"order": "asc"}}],
        },
        _source=default_includes + includes,
    )
    return {
        "embeddings": {
            hit["_id"]: hit["_source"] for hit in response["hits"]["hits"]
        },
        "total": response["hits"]["total"]["value"],
        "took": response["took"],
    }


@app.get("/prismic/{id}")
def get_prismic(
    id: str, includes: Annotated[list[allowed_includes] | None, Query()] = []
):
    """Get embeddings, filtered by prismic id"""
    response = target_es.search(
        index=f"prismic-{model_name}",
        body={
            "query": {"match": {"id": id}},
            "sort": [{"id": {"order": "asc"}}],
        },
        _source=default_includes + includes,
    )
    return {
        "embeddings": {
            hit["_id"]: hit["_source"] for hit in response["hits"]["hits"]
        },
        "total": response["hits"]["total"]["value"],
        "took": response["took"],
    }


@app.get("/embed")
def embed(query: str) -> dict:
    """Get an embedding for a supplied string"""
    embedding = model.embed(query)
    return {"embedding": embedding, "model": model_name}
