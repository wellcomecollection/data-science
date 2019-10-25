import numpy as np
import nmslib

from .aws import download_object_from_s3
from .identifiers import miro_ids, filter_invalid_ids

# download datasets
download_object_from_s3('palette/palette_embeddings.hnsw')
download_object_from_s3('palette/palette_embeddings.hnsw.dat')
palette_index = nmslib.init(method='hnsw', space='l2')
palette_index.loadIndex('palette_embeddings.hnsw', load_data=True)


def get_neighbour_ids(query_embedding, n=10, skip_first_result=False):
    '''returns n approximate nearest neigbours for a given palette embedding'''
    neighbours = palette_index.knnQueryBatch(query_embedding, n*2)
    neighbour_indexes, _ = zip(*neighbours)
    neighbour_ids = miro_ids[neighbour_indexes].tolist()
    if skip_first_result:
        neighbour_ids = neighbour_ids[1:]
    neighbour_ids = filter_invalid_ids(neighbour_ids, n)
    return neighbour_ids
