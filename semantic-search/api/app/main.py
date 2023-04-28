from pydantic import BaseModel
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
    "series"
]
allowed_works_formats = Literal[
    "Ephemera",
    "Books",
    "Pictures",
    "Archives and manuscripts",
    "Digital Images",
    "Videos",
    "Journals",
    "Audio",
]
allowed_formats = Literal[allowed_prismic_formats, allowed_works_formats]
document_types = Literal["works", "prismic"]
allowed_indexes = Literal["works", "prismic"]


class Document(BaseModel):
    format: allowed_formats
    score: Optional[float]
    id: str
    title: str
    text: Optional[str]
    type: Optional[document_types]
    embedding: Optional[list[float]]


class Response(BaseModel):
    results: dict[str, Document]
    total: int
    took: int


@app.get("/")
def health_check() -> dict:
    return {"status": "healthy"}


@app.get("/{index}")
def get(
    index: allowed_indexes,
    query: Optional[str] = None,
    n: Optional[int] = 10,
    includes: Annotated[list[allowed_includes] | None, Query()] = [],
    formats: Annotated[
        list[allowed_formats] | None,
        Query()
    ] = [],
) -> Response:
    if not query:
        return _get_all(
            index=f"{index}-{model_name}", includes=includes, formats=formats
        )
    else:
        return _get_nearest(
            index=f"{index}-{model_name}",
            query=query,
            n=n,
            includes=includes,
            formats=formats,
        )


def _get_nearest(index, query, n, includes, formats) -> Response:
    """Get nearest embeddings in an index for a supplied string"""
    embedding = model.embed(query)
    knn_query = {
        "field": "embedding",
        "query_vector": embedding,
        "k": 1000,
        "num_candidates": 1000,
    }
    if formats:
        knn_query["filter"] = {"terms": {"format": formats}}

    response = target_es.search(
        index=index,
        knn=knn_query,
        collapse={"field": "id"},
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


def _get_all(index, includes, formats) -> Response:
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
        "results": {
            hit["_id"]: hit["_source"] for hit in response["hits"]["hits"]
        },
        "total": response["hits"]["total"]["value"],
        "took": response["took"],
    }


@app.get("/{index}/{id}")
def get_by_id(
    index: allowed_indexes,
    id: str, includes: Annotated[list[allowed_includes] | None, Query()] = []
) -> Response:
    """Get embeddings, filtered by work id"""
    response = target_es.search(
        index=f"{index}-{model_name}",
        body={
            "query": {"match": {"id": id}},
            "sort": [{"id": {"order": "asc"}}],
        },
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
    embedding = model.embed(query)
    return {"embedding": embedding, "model": model_name}
