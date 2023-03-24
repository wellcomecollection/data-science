from fastapi.middleware.cors import CORSMiddleware
import os

from fastapi import FastAPI
from .src.embed import TextEmbedder
from .src.elasticsearch import get_elastic_client

target_es = get_elastic_client()
model_name = os.environ["MODEL_NAME"]
index = f"articles-{model_name}"

model = TextEmbedder(model=model_name, cache_dir="/data/embeddings")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://webapp:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/embed")
def embed(query: str) -> dict:
    """Get an embedding for a supplied string"""
    embedding = model.embed(query)
    return {"embedding": embedding, "model": model_name}


@app.get("/nearest")
def nearest(query: str, n: int = 10) -> dict:
    """Get nearest embeddings for a supplied string"""
    embedding = model.embed(query)
    response = target_es.search(
        index=index,
        knn={
            "field": "text-embedding",
            "query_vector": embedding,
            "k": n,
            "num_candidates": 1000,
        },
        collapse={"field": "id"},
    )
    return {
        "embeddings": {
            hit["_id"]: {**hit["_source"], "score": hit["_score"]}
            for hit in response["hits"]["hits"]
        },
        "total": response["hits"]["total"]["value"],
    }


@app.get("/articles")
def get_articles():
    """Get embeddings, grouped by article id"""
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
    )
    return {
        "total": response["aggregations"]["unique_ids"]["value"],
        "articles": {
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


@app.get("/articles/{id}")
def get_article(id: str):
    """Get embeddings, filtered by article id"""
    response = target_es.search(
        body={"query": {"match": {"id": id}},
              "sort": [{"id": {"order": "asc"}}]},
        index=index,
    )
    return {
        "embeddings": {hit["_id"]: hit["_source"] for hit in response["hits"]["hits"]},
        "total": response["hits"]["total"]["value"],
    }


@app.get("/embeddings")
def get_embeddings():
    """Get all embeddings"""
    response = target_es.search(index=index, body={"query": {"match_all": {}}})
    return {hit["_id"]: hit["_source"] for hit in response["hits"]["hits"]}


@app.get("/embeddings/{id}")
def get_embedding(id: str):
    """Get embedding by id"""
    response = target_es.get(index=index, id=id)
    return response["_source"]
