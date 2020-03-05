import pickle

import numpy as np

from .aws import get_object_from_s3


def encode_for_elasticsearch(clusters):
    return [f'{i}-{val}' for i, val in enumerate(clusters)]


class LSHEncoder():
    def __init__(self, model_name):
        self.models = pickle.loads(
            get_object_from_s3(f'lsh_models/{model_name}.pkl')
        )
        self.__len__ = len(self.models)

    def __call__(self, feature_vector):
        feature_groups = np.split(feature_vector, len(self))

        clusters = [
            model.predict(feature_group.reshape(1, -1))[0]
            for model, feature_group in zip(self.models, feature_groups)
        ]

        return encode_for_elasticsearch(clusters)