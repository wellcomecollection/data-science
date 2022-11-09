import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.elasticsearch import get_pipeline_elastic_client
from src.log import get_logger
from src.wellcome import images_index

log = get_logger()

data_dir = Path("/data/raw")

thumbnail_dict = {}
feature_list = []
ids = []

log.info("Loading elastic client")
es = get_pipeline_elastic_client()

batch_size = 1000
search_params = {
    "index": images_index,
    "size": batch_size,
    "query": {"match_all": {}},
    "source": [
        "display.thumbnail.url",
        "query.inferredData.features1",
        "query.inferredData.features2",
    ],
    "sort": [{"query.id": {"order": "asc"}}],
}

total_results = es.count(index=images_index)["count"]
n_batches = total_results // batch_size + 1

progress_bar = tqdm(range(n_batches))
progress_bar.set_description("Fetching initial batch")
response = es.search(search_after=[0], **search_params)
last_sort_value = response.body["hits"]["hits"][-1]["sort"]

for result in response.body["hits"]["hits"]:
    thumbnail = result["_source"]["display"]["thumbnail"]["url"]
    features = np.hstack(
        [
            result["_source"]["query"]["inferredData"]["features1"],
            result["_source"]["query"]["inferredData"]["features2"],
        ]
    )
    thumbnail_dict[result["_id"]] = thumbnail
    feature_list.append(features)
    ids.append(result["_id"])

with open(data_dir / "thumbnails.pkl", "wb") as f:
    pickle.dump(thumbnail_dict, f)

np.save(data_dir / "features.npy", np.array(feature_list))
np.save(data_dir / "ids.npy", np.array(ids))
progress_bar.update(1)

for _ in progress_bar:
    progress_bar.set_description(f"Fetching batch after {last_sort_value[0]}")
    response = es.search(search_after=last_sort_value, **search_params)
    last_sort_value = response.body["hits"]["hits"][-1]["sort"]

    for result in response.body["hits"]["hits"]:
        thumbnail = result["_source"]["display"]["thumbnail"]["url"]
        features = np.hstack(
            [
                result["_source"]["query"]["inferredData"]["features1"],
                result["_source"]["query"]["inferredData"]["features2"],
            ]
        )
        thumbnail_dict[result["_id"]] = thumbnail
        feature_list.append(features)
        ids.append(result["_id"])

    with open(data_dir / "thumbnails.pkl", "wb") as f:
        pickle.dump(thumbnail_dict, f)

    np.save(data_dir / "features.npy", np.array(feature_list))
    np.save(data_dir / "ids.npy", np.array(ids))
