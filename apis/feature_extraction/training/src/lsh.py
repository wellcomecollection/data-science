from sklearn.cluster import KMeans
import numpy as np


def split_features(feature_vectors, n_groups):
    feature_groups = np.split(
        feature_vectors,
        indices_or_sections=n_groups,
        axis=1
    )
    return feature_groups


def train_clusters(feature_group, m, sample_size=None):
    if sample_size:
        random_indexes = np.random.choice(
            len(feature_group),
            size=sample_size,
            replace=False
        )
        feature_group = feature_group[random_indexes]
    clustering_alg = KMeans(n_clusters=m).fit(feature_group)
    return clustering_alg
