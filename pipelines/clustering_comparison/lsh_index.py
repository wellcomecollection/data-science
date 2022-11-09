import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.elasticsearch import get_local_elastic_client
from src.log import get_logger
from src.lsh import encode_for_elasticsearch, split_features

log = get_logger()

base_model_dir = Path("../data/models").absolute()
model_dir = sorted(base_model_dir.glob("*"))[-1]
model_timestamp = model_dir.name

log.info("loading features and splitting them into chunks")
features = np.load("../data/features.npy")
feature_groups = split_features(features, n_groups=256)

log.info("loading kmeans models")
with open(model_dir / "kmeans.pkl", "rb") as f:
    kmeans_model_list = pickle.load(f)

log.info("predicting the kmeans clusters for all features")
kmeans_clusters = np.stack(
    [
        model.predict(feature_group)
        for model, feature_group in tqdm(
            zip(kmeans_model_list, feature_groups), desc="kmeans predict"
        )
    ],
    axis=1,
)

log.info("loading dbscan models")
with open(model_dir / "dbscan.pkl", "rb") as f:
    dbscan_model_list = pickle.load(f)

log.info("predicting the dbscan clusters for all features")
dbscan_clusters = np.stack(
    [
        model.approximate_predict(feature_group)
        for model, feature_group in tqdm(
            zip(dbscan_model_list, feature_groups), desc="dbscan predict"
        )
    ],
    axis=1,
)

log.info("loading image ids and thumbnails")
with open("../data/thumbnails.pkl", "rb") as f:
    thumbnails = pickle.load(f)

log.info("loading elastic client")
es = get_local_elastic_client()

log.info("posting the endcoded features to a dedicated elastic index")
progress = tqdm(thumbnails.items())
for i, (id, thumbnail_url) in enumerate(progress):
    progress.set_description(f"Indexing hashes for {id}")
    es.index(
        index=model_timestamp,
        id=id,
        document={
            "thumbnail_url": thumbnail_url,
            "kmeans-hash": encode_for_elasticsearch(kmeans_clusters[i]),
            "dbscan-hash": encode_for_elasticsearch(dbscan_clusters[i]),
        },
    )
