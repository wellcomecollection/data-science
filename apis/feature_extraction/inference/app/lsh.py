import pickle

import numpy as np

from .aws import get_object_from_s3


class LSHEncoder():
    def __init__(self, model_name):
        self.models = pickle.loads(
            get_object_from_s3(
                object_key=f'lsh_models/{model_name}.pkl',
                bucket_name='model-core-data',
                profile_name='data-dev'
            )
        )

    def encode_for_elasticsearch(self, clusters):
        return [f'{i}-{val}' for i, val in enumerate(clusters)]

    def __call__(self, feature_vector):
        feature_groups = np.split(feature_vector, len(self.models))

        clusters = [
            model.predict(feature_group.reshape(1, -1))[0]
            for model, feature_group in zip(self.models, feature_groups)
        ]

        return self.encode_for_elasticsearch(clusters)
