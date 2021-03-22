import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from typer import Option, Typer
from wasabi import Printer

from src.elastic import ES
from src.lsh_encoder import LSHEncoder

msg = Printer()
es = ES()

msg.warn("loading features and ids...")
feature_vectors = np.load("/data/feature_vectors.npy")
document_ids = np.load("/data/ids.npy")
msg.good("data loaded successfully")

# populate exact index with dense vectors
msg.warn("indexing exact features...")


def gendata_exact(document_ids, feature_vectors):
    loop = tqdm(
        zip(document_ids, feature_vectors),
        total=len(document_ids)
    )
    for (document_id, feature_vector) in loop:
        features_1, features_2 = feature_vector.reshape(2, 2048)
        yield {
            "_index": "feature_vectors",
            "_id": document_id,
            "features_1": features_1.tolist(),
            "features_2": features_2.tolist(),
        }


es.create_index(
    index_name="feature_vectors",
    properties={
        "features_1": {
            "type": "dense_vector",
            "dims": 2048,
        },
        "features_2": {
            "type": "dense_vector",
            "dims": 2048,
        },
    }
)

es.bulk_index_documents(
    gendata_exact(document_ids, feature_vectors)
)
msg.good("indexed exact features")


# populate LSH indexes with hashes
n_classifiers_options = [32, 64, 128, 256, 512]
n_clusters_options = [8, 16, 32, 64, 128, 256]

for n_classifiers in n_classifiers_options:
    for n_clusters in n_clusters_options:
        model_name = f"{n_classifiers}-{n_clusters}"
        msg.warn(f"training model {model_name}...")
        lsh_encoder = LSHEncoder(n_classifiers, n_clusters)
        lsh_encoder.fit(feature_vectors, n_features=10_000)
        lsh_hashes = lsh_encoder.predict(feature_vectors)
        msg.good(f"trained model {model_name}")

        msg.warn("indexing hashes...")
        es.create_index(
            index_name=model_name,
            properties={
                "lsh_features": {
                    "type": "keyword"
                }
            }
        )

        gendata_lsh = (
            {
                "_index": model_name,
                "_id": document_id,
                "lsh_features": lsh_hash
            }
            for (document_id, lsh_hash) in tqdm(
                zip(document_ids, lsh_hashes),
                total=len(document_ids)
            )
        )
        es.bulk_index_documents(gendata_lsh)
        msg.good("indexed hashes")
