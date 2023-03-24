from typing import Literal, Annotated
import os

from fastapi import FastAPI, Query
from .src.embed import TextEmbedder
from .src.elasticsearch import get_elastic_client

target_es = get_elastic_client()
model_name = os.environ["MODEL_NAME"]
index = f"prismic-{model_name}"

model = TextEmbedder(model=model_name, cache_dir="/data/embeddings")

app = FastAPI()

allowed_includes = Literal["embedding"]


@app.get("/embed")
def embed(query: str) -> dict:
    """Get an embedding for a supplied string"""
    embedding = model.embed(query)
    return {"embedding": embedding, "model": model_name}


@app.get("/nearest")
def nearest(query: str, n: int = 10, includes: Annotated[list[allowed_includes] | None, Query()] = []):
    """Get nearest embeddings for a supplied string"""
    embedding = model.embed(query)
    response = target_es.search(
        index=index,
        knn={
            "field": "embedding",
            "query_vector": embedding,
            "k": n,
            "num_candidates": 1000,
        },
        collapse={"field": "id"},
        _source=["id", "type", "title", "text", *includes],
    )
    return {
        "embeddings": {
            hit["_id"]: {**hit["_source"], "score": hit["_score"]}
            for hit in response["hits"]["hits"]
        },
        "total": response["hits"]["total"]["value"],
    }


@app.get("/documents")
def get_documents(includes: Annotated[list[allowed_includes] | None, Query()] = []):
    """Get embeddings, grouped by document id"""
    response = target_es.search(
        index=index,
        body={
            "size": 0,
            "aggs": {
                "unique_ids": {"cardinality": {"field": "id"}},
                "group_by_id": {
                    "terms": {"field": "id"},
                    "aggs": {
                        "top_hits": {
                            "top_hits": {
                                "size": 5,
                                "sort": [{"id": {"order": "asc"}}],
                            }
                        }
                    },
                },
            },
        },
        _source=["id", "type", "title", "text", *includes],
    )
    return {
        "total": response["aggregations"]["unique_ids"]["value"],
        "documents": {
            bucket["key"]: {
                "total:": bucket["doc_count"],
                "embeddings": {
                    hit["_id"]: hit["_source"]
                    for hit in bucket["top_hits"]["hits"]["hits"]
                },
            }
            for bucket in response["aggregations"]["group_by_id"]["buckets"]
        },
    }


@app.get("/documents/{id}")
def get_document(id: str, includes: Annotated[list[allowed_includes] | None, Query()] = []):
    """Get embeddings, filtered by document id"""
    response = target_es.search(
        index=index,
        body={"query": {"match": {"id": id}},
              "sort": [{"id": {"order": "asc"}}]},
        _source=["id", "type", "title", "text", *includes],
    )
    return {
        "embeddings": {hit["_id"]: hit["_source"] for hit in response["hits"]["hits"]},
        "total": response["hits"]["total"]["value"],
    }


@app.get("/embeddings")
def get_embeddings(includes: Annotated[list[allowed_includes] | None, Query()] = []):
    """Get all embeddings"""
    response = target_es.search(
        index=index,
        body={"query": {"match_all": {}}},
        _source=["id", "type", "title", "text", *includes],
    )
    return {
        "embeddings": {hit["_id"]: hit["_source"] for hit in response["hits"]["hits"]},
        "total": response["hits"]["total"]["value"],
    }


@app.get("/embeddings/{id}")
def get_embedding(id: str, includes: Annotated[list[allowed_includes] | None, Query()] = []):
    """Get embedding by id"""
    response = target_es.get(
        index=index,
        id=id,
        _source=["id", "type", "title", "text", *includes],
    )
    return response["_source"]
