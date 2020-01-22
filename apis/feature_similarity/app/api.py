import json
import re

import numpy as np
from fastapi import FastAPI, HTTPException

from .identifiers import expand_identifiers, catalogue_id_set, index_lookup
from .neighbours import feature_vectors, get_neighbour_ids

# initialise API
app = FastAPI(
    title='Feature Similarity',
    description='Find similar images based on their structural features, using 4096d embeddings from the last hidden layer of a pretrained VGG16 network',
    docs_url='/feature-similarity/docs',
    redoc_url='/feature-similarity/redoc'
)


@app.get('/works/{catalogue_id}')
def feature_similarity_by_catalogue_id(catalogue_id: str, n: int = 10):
    if catalogue_id not in catalogue_id_set:
        raise HTTPException(status_code=404, detail="Invalid catalogue id")

    query_index = index_lookup[catalogue_id]
    query_embedding = np.array(feature_vectors[query_index]).reshape(1, -1)
    neighbour_ids = get_neighbour_ids(query_embedding, n)

    return {
        'original': expand_identifiers(catalogue_id),
        'neighbours': [
            expand_identifiers(catalogue_id)
            for catalogue_id in neighbour_ids
        ]
    }


@app.get('/healthcheck')
def healthcheck():
    return {'status': 'healthy'}
