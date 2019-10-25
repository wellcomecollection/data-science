import os
import pickle

import nmslib
import numpy as np
from fastapi import FastAPI

from .utils import embed, id_to_url
from .aws import get_object_from_s3, download_object_from_s3

# Load model data (don't fetch from s3 if developing locally)
if 'DEVELOPMENT' in os.environ:
    base_path = os.path.expanduser('~/datasets/devise_search/')
    image_ids = np.load(
        os.path.join(base_path, 'image_ids.npy'),
        allow_pickle=True
    )
    search_index = nmslib.init(method='hnsw', space='cosinesimil')
    search_index.loadIndex(os.path.join(base_path, 'search_index.hnsw'))
else:
    download_object_from_s3('devise_search/search_index.hnsw')
    ids = np.load(
        get_object_from_s3('devise_search/image_ids.npy'),
        allow_pickle=True
    )
    feature_index = nmslib.init(method='hnsw', space='cosinesimil')
    feature_index.loadIndex('search_index.hnsw')


app = FastAPI(
    title='Visual-Semantic Image Search',
    description='Search Wellcome Collection\'s images based on their visual content, without making use of captions. Based on DeViSE: A Deep Visual-Semantic Embedding Model (https://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf)',
    docs_url='/devise_search/docs',
    redoc_url='/devise_search/redoc'
)


@app.get('/devise_search')
def devise_search(query_text: str = 'An old wooden boat', n: int = 10):
    query_embedding = embed(query_text)
    neighbour_indexes, _ = search_index.knnQuery(query_embedding, n)

    neighbour_ids = [image_ids[index] for index in neighbour_indexes]
    neighbour_urls = [id_to_url(id) for id in neighbour_ids]

    return {
        'query_text': query_text,
        'neighbour_ids': neighbour_ids,
        'neighbour_urls': neighbour_urls
    }


@app.get('/devise_search/health_check')
def health_check():
    return {'status': 'healthy'}
