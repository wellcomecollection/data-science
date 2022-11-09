import pickle
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from src.elasticsearch import get_local_elastic_client
from src.log import get_logger

log = get_logger()

data_path = Path("/data/raw").absolute()

features = np.load(data_path / "features.npy")


# load the thumbnail and image id data
# they're numpy arrays, so we can use them to index into the distances matrix
ids = np.load(data_path / "ids.npy")
thumbnails = np.load(data_path / "thumbnails.npy")

# choose a random image
i = np.random.randint(0, features.shape[0])
query_image_id = ids[i]
query_image_thumbnail = thumbnails[i].replace(
    "/info.json", "/full/!200,200/0/default.jpg"
)
print("Query image:")
print(f"   {query_image_id}  |  {query_image_thumbnail}\n")

# calculate and sort the distances
distances = cdist(features[i].reshape(1, -1), features, metric="cosine")[0]
sorted_distances = np.argsort(distances)

top_10_ids = ids[sorted_distances[:10]]
top_10_thumbnails = thumbnails[sorted_distances[:10]]

# print the image id and thumbnail for each of the top 10 images
print("10 exact closest images:")
for i, (image_id, thumbnail) in enumerate(zip(top_10_ids, top_10_thumbnails)):
    thumbnail = thumbnail.replace("/info.json", "/full/!200,200/0/default.jpg")
    print(f"{i}. {image_id}  |  {thumbnail}")


# use the elasticsearch client rank eval API to check the performance of the
# knn query
es = get_local_elastic_client()
index_name = "images-knn-256"

# get the query features from the elasticsearch index
query_features = es.get(
    index=index_name,
    id=query_image_id,
    source="features",
)["_source"]["features"].tolist()

print(query_features)

rank_eval_response = es.rank_eval(
    index=index_name,
    requests=[
        {
            "id": query_image_id,
            "request": {
                "knn": {
                    "field": "features",
                    "query_vector": query_features,
                    "k": 10,
                    "num_candidates": 100,
                },
                "fields": ["thumbnail-url"],
            },
            "rating": {
                "relevant": [
                    {
                        "_id": id,
                        "_index": index_name,
                    }
                    for id in top_10_ids
                ]
            },
        }
    ],
    metric={
        "precision": {
            "relevant_rating_threshold": 1,
            "k": 10,
        }
    },
)

print(json.dumps(rank_eval_response, indent=2))
