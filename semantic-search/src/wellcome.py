from typing import Generator
from .elasticsearch import get_pipeline_es_client, get_reporting_es_client
import httpx
from .log import get_logger

log = get_logger()

api_url = "https://api.wellcomecollection.org/catalogue/v2/works"
works_index_name = httpx.get(
    "https://api.wellcomecollection.org/catalogue/v2/_elasticConfig"
).json()["worksIndex"]


def count_works() -> int:
    response = httpx.get(api_url).json()
    return response["totalResults"]


def yield_works(batch_size: int) -> Generator[dict, None, None]:
    response = httpx.get(
        api_url,
        params={"pageSize": batch_size},
    ).json()

    while True:
        for result in response["results"]:
            yield result
        if response["nextPage"] is None:
            break
        else:
            response = httpx.get(response["nextPage"]).json()


def yield_popular_works(size: int = 10_000) -> Generator[dict, None, None]:
    log.debug("Fetching popular work IDs")
    reporting_es_client = get_reporting_es_client()
    response = reporting_es_client.search(
        index="metrics-conversion-prod",
        size=0,
        query={
            "bool": {
                "must": [
                    {"term": {"page.name": {"value": "work"}}},
                    {"range": {"@timestamp": {"gte": "2021-09-01"}}},
                ]
            }
        },
        aggs={"popular_works": {"terms": {"field": "page.query.id", "size": size}}},
    )

    popular_work_ids = [
        bucket["key"] for bucket in response["aggregations"]["popular_works"]["buckets"]
    ]

    log.debug(f"Found popular work IDs: {popular_work_ids}")

    log.debug("Fetching popular works")
    pipeline_es_client = get_pipeline_es_client()
    for work_id in popular_work_ids:
        log.debug(f"Fetching work {work_id}")
        try:
            document = pipeline_es_client.get(
                index=works_index_name,
                id=work_id
            )
            work = document["_source"]["display"]
            yield work
        except KeyError as key:
            log.debug(f"Could not fetch work {work_id}: missing key {key}")
