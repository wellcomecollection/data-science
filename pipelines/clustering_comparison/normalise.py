import numpy as np
from elastic_transport import ConnectionTimeout
from tqdm import tqdm

from src.elasticsearch import get_local_elastic_client
from src.log import get_logger

log = get_logger()

log.info("loading elastic client")
es = get_local_elastic_client()

dim = 256
source_index_name = f"images-umap-{dim}"
target_index_name = f"{source_index_name}-normalised"

log.info(f"create index {target_index_name}")
es.indices.delete(
    index=target_index_name,
    ignore_unavailable=True,
)
es.indices.create(
    index=target_index_name,
    mappings={
        "properties": {
            "features": {
                "type": "dense_vector",
                "dims": dim,
                "index": True,
                "similarity": "dot_product",
            },
            "thumbnail-url": {"type": "keyword"},
        }
    },
)


batch_size = 100
search_params = {
    "index": source_index_name,
    "size": batch_size,
    "query": {"match_all": {}},
    "source": [
        "thumbnail-url",
        "features",
    ],
    "sort": [{"thumbnail-url": {"order": "asc"}}],
}

total_results = es.count(index=source_index_name)["count"]
n_batches = total_results // batch_size + 1

progress_bar = tqdm(range(n_batches))
progress_bar.set_description("Fetching initial batch")
response = es.search(search_after=[0], **search_params)
last_sort_value = response.body["hits"]["hits"][-1]["sort"]

for i in progress_bar:
    progress_bar.set_description(f"Normalising batch {i}")
    response = es.search(search_after=last_sort_value, **search_params)
    hits = response.body["hits"]["hits"]
    if len(hits) == 0:
        break
    last_sort_value = hits[-1]["sort"]

    # normalise the vectors to have unit length
    vectors = np.array([hit["_source"]["features"] for hit in hits])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # post the normalised vectors to the new index
    for hit, vector in zip(hits, vectors):
        es.index(
            index=target_index_name,
            id=hit["_id"],
            document={
                "thumbnail-url": hit["_source"]["thumbnail-url"],
                "features": vector.tolist(),
            },
        )
