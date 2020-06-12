from pathlib import Path

import numpy as np
from tqdm import tqdm
from typer import Option, Typer

from src.elastic import ES
from src.lsh_encoder import LSHEncoder

"""
Populate elasticsearch with feature vectors and corresponding LSH hashes
"""

es = ES()
cli = Typer(help=__doc__, no_args_is_help=True)
data_path = Path("/data")


@cli.command()
def train(
    feature_path: Path = Option(
        default=data_path/"feature_vectors.npy",
        help="path to the feature vector data to train on"
    ),
    id_path: Path = Option(
        default=data_path/"ids.npy",
        help="path to the ids of the feature vectors to train on"
    ),
    n_classifiers: int = Option(
        ..., help="The number of classifiers to train, ie the number of chunks the feature vectors will be split into"
    ),
    n_clusters: int = Option(
        ..., help="The number of clusters to find with each classifier"
    ),
    n_features: int = Option(
        default=0,
        help="The number features to train on and encode"
    ),
    save: bool = Option(False, "--save")
):
    feature_vectors = np.load("/data/feature_vectors.npy")
    document_ids = np.load("/data/ids.npy")
    lsh_encoder = LSHEncoder(n_classifiers, n_clusters)
    lsh_encoder.fit(feature_vectors, n_features)
    lsh_hashes = lsh_encoder.predict(feature_vectors)

    model_name = f"{n_classifiers}-{n_clusters}"
    if save:
        lsh_encoder.save(Path(f"./data/models/{model_name}.pkl"))
        print("Saving encoded features")
        with open(str(Path(f"./data/encoded/{model_name}.npy")), "wb") as f:
            np.save(f, lsh_hashes)

    es.create_index(
        index_name=model_name,
        properties={
            "lsh_features": {
                "type": "keyword"
            }
        }
    )

    gendata = (
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
    es.bulk_index_documents(gendata)


if __name__ == "__main__":
    cli()
