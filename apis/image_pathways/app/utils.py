import os

import nmslib
import numpy as np

from .aws import download_object_from_s3, get_object_from_s3

# Load model data (don't fetch from s3 if developing locally)
if 'DEVELOPMENT' in os.environ:
    base_path = os.path.expanduser('~/datasets/image_pathways')
    feature_index = nmslib.init(method='hnsw', space='cosinesimil')
    feature_index.loadIndex(
        os.path.join(base_path, 'feature_vectors.hnsw'), load_data=True
    )
    ids = np.load(os.path.join(base_path, 'ids.npy'))
else:
    download_object_from_s3('image_pathways/feature_vectors.hnsw')
    download_object_from_s3('image_pathways/feature_vectors.hnsw.dat')
    feature_index = nmslib.init(method='hnsw', space='cosinesimil')
    feature_index.loadIndex('feature_vectors.hnsw', load_data=True)
    ids = np.load(get_object_from_s3('image_pathways/ids.npy'))


def get_random_image_id(excluding=None):
    if excluding:
        random_id = np.random.choice(np.setdiff1d(ids, excluding))
    else:
        random_id = np.random.choice(ids)
    return random_id


def get_ideal_coords(start_coord, end_coord, n):
    '''returns n evenly spaced points between start_coord and end_coord'''
    line = end_coord - start_coord
    linspace = np.linspace(0, 1, n)[1:-1]  # n points between 0 and 1
    ideal_coords = np.array([(x * line) + start_coord for x in linspace])
    return ideal_coords


def get_path_indexes(closest_indexes, start_index, end_index):
    '''
    Return the indexes of the points which most closely match an ideal path
    '''
    path_indexes = []
    for row in closest_indexes:
        i, found_node = 0, False
        while not found_node:
            index_is_valid = (
                (row[i] not in path_indexes) and
                (row[i] not in [start_index, end_index])
            )
            if index_is_valid:
                path_indexes.append(row[i])
                found_node = True
            else:
                i += 1

    return [start_index] + path_indexes + [end_index]


def get_pathway(id_1, id_2, n_nodes, ids=ids, feature_index=feature_index):
    # Get the start and end coordinates
    start_index = np.where(ids == id_1)[0][0]
    end_index = np.where(ids == id_2)[0][0]
    start_coord = np.array(feature_index[start_index])
    end_coord = np.array(feature_index[end_index])

    # Get n points along the ideal line between start_coord and end_coord
    ideal_coords = get_ideal_coords(start_coord, end_coord, n_nodes)

    # Get the indexes of the 10 closest images to each point on the ideal path.
    # 100 closest ids should be more than enough to avoid conflicts even for
    # large requests, and minimises the load on the feature_index and the amount
    # of data we need to pass around.
    neighbours = feature_index.knnQueryBatch(ideal_coords, 100)
    closest_indexes, _ = zip(*neighbours)
    closest_indexes = np.stack(closest_indexes)

    # Add the best set of images to the path
    path_indexes = get_path_indexes(closest_indexes, start_index, end_index)
    pathway = [ids[index] for index in path_indexes]
    return pathway


def id_to_url(image_id):
    return f'https://iiif.wellcomecollection.org/image/{image_id}.jpg/full/960,/0/default.jpg'
