from .elasticsearch import get_catalogue_elastic_client


def yield_source_images(pipeline_date):
    es = get_catalogue_elastic_client(pipeline_date)
    index_name = f"images-indexed-{pipeline_date}"

    pit = es.open_point_in_time(index=index_name, keep_alive="12h")
    search_after = None
    while True:
        results = es.search(
            body={
                "size": 100,
                "query": {"match_all": {}},
                "_source": ["query.id", "display"],
                "sort": [{"query.id": "asc"}],
                "pit": {"id": pit["id"], "keep_alive": "1m"},
                "search_after": search_after
            },
        )

        for hit in results["hits"]["hits"]:
            yield hit["_source"]["display"]
        if len(results["hits"]["hits"]) < 100:
            break

        search_after = [results["hits"]["hits"][-1]["_source"]["query"]["id"]]
    es.close_point_in_time(id=pit["id"])


def count_source_images(pipeline_date):
    es = get_catalogue_elastic_client(pipeline_date)
    index_name = f"images-indexed-{pipeline_date}"
    return es.count(index=index_name)["count"]
