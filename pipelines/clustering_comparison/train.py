import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import typer
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from tqdm import tqdm
from umap import UMAP

from src.log import get_logger
from src.lsh import select_n_random_feature_vectors, split_features

log = get_logger()
app = typer.Typer()


@app.command()
def train():
    timestamp = datetime.now().isoformat(timespec="seconds")

    train_kmeans = typer.confirm(
        "Do you want to train a set of kmeans models?", default=True
    )
    train_dbscan = typer.confirm(
        "Do you want to train a set of dbscan models?", default=True
    )
    train_umap = typer.confirm(
        "Do you want to train a UMAP model?", default=True
    )

    data_path = Path("../data").absolute()
    model_path = data_path / "models"
    model_path.mkdir(parents=True, exist_ok=True)

    log.info("Loading features")
    features = np.load(data_path / "raw" / "features.npy")

    sample_size = typer.prompt(
        "How large should the training sample be?", default=10_000
    )
    subset_features = select_n_random_feature_vectors(features, n=sample_size)

    if train_kmeans or train_dbscan:
        n_sub_models = typer.prompt(
            "How many sub-models should the kmeans/dbscan model(s) contain?",
            default=256,
        )
        log.info(f"Picking a subset of {sample_size} features for training")
        feature_groups = split_features(subset_features, n_groups=n_sub_models)

        if train_kmeans:
            log.info("Training kmeans models")
            n_clusters = typer.prompt(
                "How many clusters should each model contain?", default=256
            )
            kmeans_model_list = []
            for feature_group in tqdm(feature_groups, desc="kmeans training"):
                clusterer = KMeans(n_clusters=n_clusters).fit(feature_group)
                kmeans_model_list.append(clusterer)

            log.info("Saving kmeans models")
            with open(model_path / "kmeans" / f"{timestamp}.pkl", "wb") as f:
                pickle.dump(kmeans_model_list, f)

        if train_dbscan:
            log.info("Training dbscan models")
            dbscan_model_list = []
            for feature_group in tqdm(feature_groups, desc="dbscan training"):
                clusterer = HDBSCAN(min_cluster_size=10).fit(feature_group)
                dbscan_model_list.append(clusterer)

            log.info("Saving dbscan models")
            with open(model_path / "dbscan" / f"{timestamp}.pkl", "wb") as f:
                pickle.dump(dbscan_model_list, f)

    if train_umap:
        log.info("Training UMAP model")
        n_components = typer.prompt(
            "How many dimensions should the UMAP model reduce to?"
        )
        reducer = UMAP(n_components=int(n_components), metric="cosine").fit(
            subset_features
        )

        log.info("Saving umap model")
        with open(model_path / "umap" / f"{timestamp}.pkl", "wb") as f:
            pickle.dump(reducer, f)

    log.info("Done")


if __name__ == "__main__":
    app()
