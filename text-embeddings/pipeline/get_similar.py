import os

from src.elasticsearch import get_elastic_client
from src.log import get_logger

log = get_logger()

es = get_elastic_client()
index_name = os.environ.get("ES_INDEX")
field_name = "title_embedding"

query_data = es.search(
    index=index_name, query={"function_score": {"random_score": {}}}, size=1
)["hits"]["hits"][0]

log.info("Random query story:")
log.info(query_data["_source"]["title"])
log.info(query_data["_id"])
log.info(query_data["_source"]["standfirst"] + "\n")

knn_response = es.knn_search(
    index=index_name,
    knn={
        "field": field_name,
        "query_vector": query_data["_source"][field_name],
        "k": 7,
        "num_candidates": 100,
    },
)

log.info("Similar stories:")
for i, hit in enumerate(knn_response["hits"]["hits"][1:]):
    print(f'{i+1}. ({hit["_score"]: .2f})  {hit["_id"]}')
    print(hit["_source"]["title"])
    print(hit["_source"]["standfirst"] + "\n")
