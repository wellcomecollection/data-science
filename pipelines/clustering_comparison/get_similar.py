from src.elasticsearch import get_local_elastic_client, get_rank_elastic_client

es = get_rank_elastic_client()

# index_name = "images-knn"
index_name = "images-knn-1024"
field_name = "features"

# get a random document from the index
query_data = es.search(
    index=index_name,
    query={"function_score": {"random_score": {}}},
    size=1,
)["hits"]["hits"][0]

print(f"query id: {query_data['_id']}")

query_url = query_data["_source"]["thumbnail-url"].replace(
    "info.json", "full/400,/0/default.jpg"
)
print(f"query image: {query_url}")

knn_response = es.knn_search(
    index=index_name,
    knn={
        "field": field_name,
        "query_vector": query_data["_source"][field_name],
        "k": 10,
        "num_candidates": 10,
    },
)

hits = knn_response["hits"]["hits"]

print("similar images:")
for i, hit in enumerate(hits[1:]):
    url = hit["_source"]["thumbnail-url"].replace(
        "info.json", "full/400,/0/default.jpg"
    )
    score = hit["_score"]
    print(f"{i+1}. {url} ({score})")

print(f"took: {knn_response['took']} ms")
