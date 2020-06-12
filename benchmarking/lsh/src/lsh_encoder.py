import os
import pickle

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


class LSHEncoder:
    def __init__(self, n_classifiers, n_clusters):
        self.n_classifiers = n_classifiers
        self.n_clusters = n_clusters
        self.models = [
            KMeans(n_clusters=self.n_clusters)
            for _ in range(self.n_classifiers)
        ]

    def __call__(self, feature_vectors):
        return self.predict(feature_vectors)

    def load(self, model_path):
        with open(model_path, "rb") as f:
            self.models = pickle.load(f)

    def fit(self, feature_vectors, n_features=None):
        if n_features:
            print(
                f"Training on {n_features} features "
                f"({len(feature_vectors)} available)"
            )
            sample_indexes = np.random.choice(
                feature_vectors.shape[0]-1, size=n_features, replace=False
            )
            feature_vectors = feature_vectors[sample_indexes]

        feature_groups = np.split(
            feature_vectors,
            indices_or_sections=self.n_classifiers,
            axis=1
        )
        train_pairs = list(zip(self.models, feature_groups))
        for i, (model, data) in enumerate(tqdm(train_pairs)):
            self.models[i] = model.fit(data)

    def encode_for_elasticsearch(self, clusters):
        return [f"{i}-{val}" for i, val in enumerate(clusters)]

    def predict(self, feature_vectors):
        print("Encoding LSH hashes")
        feature_groups = np.split(
            feature_vectors,
            indices_or_sections=self.n_classifiers,
            axis=1
        )

        clusters = np.stack(
            [
                model.predict(data)
                for model, data in zip(self.models, feature_groups)
            ],
            axis=1,
        )

        return [self.encode_for_elasticsearch(c) for c in clusters]

    def save(self, model_path):
        print("Saving model")
        with open(str(model_path), "wb") as f:
            pickle.dump(self.models, f)
