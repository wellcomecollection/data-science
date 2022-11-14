from typing import Generator

from elasticsearch import Elasticsearch


def yield_concepts(
    es: Elasticsearch, index: str, batch_size: int, limit: int
) -> Generator[dict, None, None]:
    search_params = {
        "index": index,
        "size": batch_size,
        "query": {
            "bool": {
                "should": {"match_all": {}},
            }
        },
        "source": ["query.label"],
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
                if i >= limit:
                    return
        except IndexError:
            break
