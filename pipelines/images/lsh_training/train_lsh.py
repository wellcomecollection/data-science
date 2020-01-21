import os
import pickle

import click
import numpy as np
from tqdm import tqdm

from src.aws import get_keys_and_ids, get_object_from_s3
from src.lsh import split_features, train_clusters


def get_feature_vectors_from_s3():
    '''This is unbearably slow. Don't do it'''
    keys, _, _ = get_keys_and_ids(
        bucket_name='miro-images-feature-vectors',
        profile_name='data-dev',
        prefix='feature_vectors'
    )
    print('Fetching feature vectors')
    feature_vectors = np.stack([
        np.frombuffer(get_object_from_s3(
            object_key=key,
            bucket_name='miro-images-feature-vectors',
            profile_name='data-dev'
        ), dtype=np.float32)
        for key in tqdm(keys)
    ])
    return np.stack(feature_vectors)


def load_local_feature_vectors(feature_vector_path):
    feature_vectors = []
    for miro_id in tqdm(os.listdir(feature_vector_path)):
        with open(os.path.join(feature_vector_path, miro_id)) as f:
            feature_vectors.append(np.fromfile(f, dtype=np.float32))
    return np.stack(feature_vectors)


@click.command()
@click.option('-n', help='number of groups to split the feature vectors into', default=256)
@click.option('-m', help='number of clusters to find within each feature group', default=32)
@click.option('--sample_size', help='number of embeddings to train clusters on', default=25_000)
@click.option('--feature_vector_path', help='path to a synced local version of the fvs in s3')
def main(n, m, sample_size, feature_vector_path):
    if not os.path.exists('models'):
        os.mkdir('models')

    if feature_vector_path:
        feature_vectors = load_local_feature_vectors(feature_vector_path)
    else:
        feature_vectors = get_feature_vectors_from_s3()

    feature_groups = split_features(feature_vectors, n)
    for i, feature_group in enumerate(tqdm(feature_groups)):
        clustering_alg = train_clusters(feature_group, sample_size=sample_size)
        padded_i = str(i).zfill(4)
        with open(f'models/group_{padded_i}.pkl', 'wb') as f:
            pickle.dump(clustering_alg, f)


if __name__ == "__main__":
    main()
