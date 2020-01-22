import os
import nmslib

from .aws import download_object_from_s3
from .identifiers import catalogue_ids

# download datasets
file_name = 'feature_vectors.hnsw'
if not os.path.exists(file_name):
    download_object_from_s3(f'feature-similarity/2020-01-22/{file_name}')
    download_object_from_s3(f'feature-similarity/2020-01-22/{file_name}.dat')

feature_vectors = nmslib.init(method='hnsw', space='cosinesimil')
feature_vectors.loadIndex(file_name, load_data=True)


def get_neighbour_ids(query_embedding, n=10, skip_first_result=True):
    '''returns n approximate nearest neigbours for a given feature vector'''
    neighbours = feature_vectors.knnQueryBatch(query_embedding, n+1)
    neighbour_indexes, _ = zip(*neighbours)
    neighbour_ids = catalogue_ids[neighbour_indexes].tolist()
    if skip_first_result:
        neighbour_ids = neighbour_ids[1:]

    return neighbour_ids
