import os
import nmslib

from .aws import download_object_from_s3
from .identifiers import miro_ids_in_nmslib_order, filter_invalid_ids

# download datasets
file_name = 'feature_vectors.hnsw'
if not os.path.exists(file_name):
    download_object_from_s3(f'image_pathways/{file_name}')
    download_object_from_s3(f'image_pathways/{file_name}.dat')

feature_vectors = nmslib.init(method='hnsw', space='cosinesimil')
feature_vectors.loadIndex('feature_vectors.hnsw', load_data=True)


def get_neighbour_ids(query_embedding, n=10, skip_first_result=False):
    '''returns n approximate nearest neigbours for a given feature vector'''
    neighbours = feature_vectors.knnQueryBatch(query_embedding, n*2)
    neighbour_indexes, _ = zip(*neighbours)
    neighbour_ids = miro_ids_in_nmslib_order[neighbour_indexes].tolist()
    if skip_first_result:
        neighbour_ids = neighbour_ids[1:]
    neighbour_ids = filter_invalid_ids(neighbour_ids, n)
    return neighbour_ids
