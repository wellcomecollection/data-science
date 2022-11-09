import numpy as np


def split_features(feature_vectors, n_groups):
    feature_groups = np.split(
        feature_vectors, indices_or_sections=n_groups, axis=1
    )
    return feature_groups


def select_n_random_feature_vectors(feature_vectors, n):
    random_indices = np.random.choice(
        feature_vectors.shape[0], n, replace=False
    )
    return feature_vectors[random_indices]


def encode_for_elasticsearch(clusters):
    encoded = []
    for i, val in enumerate(clusters):
        if val != -1:
            encoded.append(f"{i}-{val}")
    return encoded
