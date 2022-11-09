from pathlib import Path

import numpy as np
from elastic_transport import ConnectionTimeout
from tqdm import tqdm

from src.elasticsearch import get_local_elastic_client, get_rank_elastic_client
from src.log import get_logger
from src.lsh import split_features

log = get_logger()

data_path = Path("/data/raw")

log.info("load features")

features = np.load(data_path / "features.npy")[:, :1024]
features = features / np.linalg.norm(features, axis=1, keepdims=True)


log.info("load ids and thumbnail urls")
image_ids = np.load(data_path / "ids.npy")
thumbnails = np.load(data_path / "thumbnails.npy")

log.info("set up elastic client")
es = get_rank_elastic_client()

index_name = "images-knn-1024"
log.info("create index")
es.indices.delete(
    index=index_name, ignore_unavailable=True, allow_no_indices=True
)

# es.indices.create(
#     index=index_name,
#     mappings={
#         "properties": {
#             "features-1": {
#                 "type": "dense_vector",
#                 "dims": 1024,
#                 "index": True,
#                 "similarity": "cosine",
#             },
#             "features-2": {
#                 "type": "dense_vector",
#                 "dims": 1024,
#                 "index": True,
#                 "similarity": "cosine",
#             },
#             "features-3": {
#                 "type": "dense_vector",
#                 "dims": 1024,
#                 "index": True,
#                 "similarity": "cosine",
#             },
#             "features-4": {
#                 "type": "dense_vector",
#                 "dims": 1024,
#                 "index": True,
#                 "similarity": "cosine",
#             },
#             "thumbnail-url": {"type": "keyword"},
#         }
#     },
# )

es.indices.create(
    index=index_name,
    mappings={
        "properties": {
            "features": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "dot_product",
            },
            "thumbnail-url": {"type": "keyword"},
        }
    },
)

log.info("Index documents")
progress_bar = tqdm(enumerate(zip(
    image_ids,
    thumbnails,
    features,
)), total=len(image_ids))
for i, (image_id, thumbnail, feature) in progress_bar:
    progress_bar.set_description(f"Indexing {image_id}")
    try:
        es.index(
            index=index_name,
            id=image_id,
            document={
                "features": features[i],
                "thumbnail-url": thumbnail,
            },
        )
    except ConnectionTimeout:
        es = get_local_elastic_client()
        es.index(
            index=index_name,
            id=image_id,
            document={
                "features": features[i],
                "thumbnail-url": thumbnail,
            },
        )
    except Exception as e:
        log.error(f"Error indexing {image_id}: {e}")
