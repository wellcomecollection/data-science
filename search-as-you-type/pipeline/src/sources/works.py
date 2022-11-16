from typing import Generator

from elasticsearch import Elasticsearch

query = {
    "bool": {
        "should": [{"match_all": {}}],
        "filter": [{"term": {"type": "Visible"}}],
    }
}

random_sort_query = {
    "function_score": {
        "query": query,
        "functions": [{"random_score": {}}],
        "boost_mode": "replace",
    }
}


def yield_works(
    es: Elasticsearch, index: str, batch_size: int, limit: int
) -> Generator[dict, None, None]:
    search_params = {
        "index": index,
        "size": batch_size,
        "query": random_sort_query,
        "source": [
            "display.title",
            "display.contributors",
            "display.thumbnail",
        ],
        "sort": [{"query.id": {"order": "asc"}}],
        "search_after": [0],
    }
    i = 0
    while True:
        try:
            hits = es.search(**search_params).body["hits"]["hits"]
            search_params["search_after"] = hits[-1]["sort"]
            for result in hits:
                i += 1
                yield result
                if limit:
                    if i >= limit:
                        return
        except IndexError:
            break


def count_works(es: Elasticsearch, index: str) -> int:
    return es.count(index=index, query=query)["count"]
