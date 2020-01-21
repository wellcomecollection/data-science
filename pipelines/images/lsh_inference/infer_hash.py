import os
import pickle
from io import BytesIO

import click
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.aws import get_object_from_s3


def listify_for_es(cluster_array):
    return [f'{i}-{val}' for i, val in enumerate(cluster_array)]


@click.command()
@click.option('--feature_vector_s3_key', '-k')
@click.option('--model_dir')
def main(feature_vector_s3_key, model_dir):
    models = []
    for model_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_name)
        with open(model_path, 'rb') as f:
            models.append(pickle.load(f))

    feature_vector = np.frombuffer(get_object_from_s3(
        object_key=feature_vector_s3_key,
        bucket_name='miro-images-feature-vectors',
        profile_name='data-dev'
    ), dtype=np.float32)

    feature_groups = np.split(feature_vector, len(models))

    clusters = [
        model.predict(feature_group.reshape(1, -1))[0]
        for model, feature_group in zip(models, feature_groups)
    ]

    print(listify_for_es(clusters))


if __name__ == "__main__":
    main()
