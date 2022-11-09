import pickle
from pathlib import Path

import numpy as np
import typer
from elastic_transport import ConnectionTimeout
from tqdm import tqdm

from src.elasticsearch import get_local_elastic_client
from src.log import get_logger

log = get_logger()


data_dir = Path("/data")
model_dir = data_dir / "models" / "umap"
model_name = sorted(model_dir.glob("*"))[-1]

log.info(f"loading UMAP model: {model_name}")
with open(model_name, "rb") as f:
    reducer = pickle.load(f)

dim = reducer.n_components

log.info("loading elastic client")
es = get_local_elastic_client()

index_name = f"images-umap-{dim}"

log.info(f"create index {index_name}")
es.indices.delete(
    index=index_name,
    ignore_unavailable=True,
)
es.indices.create(
    index=index_name,
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

log.info("loading features")
features = np.load(data_dir / "raw" / "features.npy")
n_images = features.shape[0]

log.info("loading image ids and thumbnails")
with open(data_dir / "raw" / "thumbnails.pkl", "rb") as f:
    thumbnails = pickle.load(f)

features_batches = np.array_split(features, n_images // 100)
del features  # free up memory

i = 0
progress_bar = tqdm(features_batches, total=n_images)
for batch in progress_bar:
    progress_bar.set_description(f"Reducing batch {i}-{i+len(batch)}")
    reduced_batch = reducer.transform(batch)
    reduced_batch /= np.linalg.norm(reduced_batch, axis=1, keepdims=True)

    ids, thumnail_urls = zip(*list(thumbnails.items())[i : i + len(batch)])
    for j in range(len(batch)):
        progress_bar.set_description(f"Indexing {ids[j]}")
        try:
            es.index(
                index=index_name,
                id=ids[j],
                document={
                    "features": reduced_batch[j].tolist(),
                    "thumbnail-url": thumnail_urls[j],
                },
            )
        except ConnectionTimeout:
            es = get_local_elastic_client()
            es.index(
                index=index_name,
                id=ids[j],
                document={
                    "features": reduced_batch[j].tolist(),
                    "thumbnail-url": thumnail_urls[j],
                },
            )
        except Exception as e:
            log.error(f"Error indexing {ids[j]}: {e}")
        progress_bar.update(1)
    i += len(batch)
