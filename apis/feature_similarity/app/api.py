import json
import re

import numpy as np
from fastapi import FastAPI, HTTPException

from .identifiers import (catalogue_id_to_miro_id, index_lookup,
                          miro_id_to_identifiers, valid_catalogue_ids)
from .neighbours import feature_index, get_neighbour_ids

# initialise API
app = FastAPI(
    title='Feature Similarity',
    description='Find similar images based on their structural features, using 4096d embeddings from the last hidden layer of a pretrained VGG16 network',
    docs_url='/feature-similarity/docs',
    redoc_url='/feature-similarity/redoc'
)


@app.get('/works/{catalogue_id}')
def feature_similarity_by_catalogue_id(catalogue_id: str, n: int = 10):
    catalogue_id = catalogue_id or np.random.choice(valid_catalogue_ids)
    if catalogue_id not in valid_catalogue_ids:
        raise HTTPException(status_code=404, detail="Invalid catalogue id")

    miro_id = catalogue_id_to_miro_id[catalogue_id]
    query_index = index_lookup[miro_id]
    query_embedding = np.array(feature_index[query_index]).reshape(1, -1)
    neighbour_ids = get_neighbour_ids(
        query_embedding, n, skip_first_result=True)

    return {
        'original': miro_id_to_identifiers(miro_id),
        'neighbours': [
            miro_id_to_identifiers(miro_id)
            for miro_id in neighbour_ids
        ]
    }


@app.get('/healthcheck')
def healthcheck():
    return {'status': 'healthy'}
