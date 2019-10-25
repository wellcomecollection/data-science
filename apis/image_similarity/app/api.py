import os

import nmslib
import numpy as np
from fastapi import FastAPI, HTTPException

from .aws import download_object_from_s3, get_object_from_s3

app = FastAPI(
    title='Image Similarity',
    description='Find similar images based on their visual similarity, using VGG16 feature vectors',
    docs_url='/image_similarity/docs',
    redoc_url='/image_similarity/redoc'
)

# Load model data (don't fetch from s3 if developing locally)
if 'DEVELOPMENT' in os.environ:
    base_path = os.path.expanduser('~/datasets/image_pathways/')
    feature_index = nmslib.init(method='hnsw', space='cosinesimil')
    feature_index.loadIndex(
        os.path.join(base_path, 'feature_vectors.hnsw'),
        load_data=True
    )
    ids = np.load(os.path.join(base_path, 'ids.npy'))
else:
    download_object_from_s3('image_pathways/feature_vectors.hnsw')
    download_object_from_s3('image_pathways/feature_vectors.hnsw.dat')
    feature_index = nmslib.init(method='hnsw', space='cosinesimil')
    feature_index.loadIndex('feature_vectors.hnsw', load_data=True)
    ids = np.load(get_object_from_s3('image_pathways/ids.npy'))


def id_to_url(image_id):
    return f'https://iiif.wellcomecollection.org/image/{image_id}.jpg/full/960,/0/default.jpg'


@app.get('/image_similarity')
def image_similarity(image_id: str = None, n: int = 10):
    image_id = image_id or np.random.choice(ids)
    if image_id not in ids:
        raise HTTPException(status_code=404, detail="Invalid image_id")

    ideal_coord = np.array([feature_index[
        np.where(ids == image_id)[0][0]
    ]])

    neighbours = feature_index.knnQueryBatch(ideal_coord, n+1)
    neighbour_indexes, _ = zip(*neighbours)

    neighbour_ids = [ids[index] for index in neighbour_indexes[0]][1:]
    neighbour_urls = [id_to_url(image_id) for image_id in neighbour_ids]

    return {
        'original_image_id': image_id,
        'original_image_url': id_to_url(image_id),
        'neighbour_ids': neighbour_ids,
        'neighbour_urls': neighbour_urls
    }


@app.get('/image_similarity/health_check')
def health_check():
    return {'status': 'healthy'}
