import json
import re

import numpy as np
from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware

from .colours import random_hex
from .identifiers import (catalogue_id_to_miro_id, valid_catalogue_ids,
                          miro_id_to_identifiers, index_lookup)
from .neighbours import get_neighbour_ids, palette_index
from .palette_embedder import embed_hex_palette

# initialise API
app = FastAPI(
    title='Palette Similarity',
    description='Find similar images based on their colour, using approximate embeddings of euclidean distance in LAB space between 5-colour palettes',
    docs_url='/palette-api/docs',
    redoc_url='/palette-api/redoc'
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)


# create API endpoints


@app.get('/works/{catalogue_id}')
def palette_similarity_by_catalogue_id(catalogue_id: str, n: int = 10):
    catalogue_id = catalogue_id or np.random.choice(valid_catalogue_ids)
    if catalogue_id not in valid_catalogue_ids:
        raise HTTPException(status_code=404, detail="Invalid catalogue id")

    miro_id = catalogue_id_to_miro_id[catalogue_id]
    query_index = index_lookup[miro_id]
    query_embedding = np.array(palette_index[query_index]).reshape(1, -1)
    neighbour_ids = get_neighbour_ids(
        query_embedding, n, skip_first_result=True)

    return {
        'original': miro_id_to_identifiers(miro_id),
        'neighbours': [
            miro_id_to_identifiers(miro_id)
            for miro_id in neighbour_ids
        ]
    }


@app.get('/palette')
def palette_similarity_by_palette(palette: list = None, n: int = 10):
    if palette:
        palette = json.loads(palette)
    else:
        palette = [random_hex() for _ in range(5)]

    if len(palette) != 5:
        raise HTTPException(
            status_code=422,
            detail='Palette must consist of 5 colours'
        )

    for colour in palette:
        if not re.fullmatch(r'[A-Fa-f0-9]{6}', colour):
            raise HTTPException(
                status_code=422,
                detail=f'{colour} is not a valid hex colour'
            )

    query_embedding = embed_hex_palette(palette)
    neighbour_ids = get_neighbour_ids(query_embedding, n)

    return {
        'original': {
            'palette': palette,
        },
        'neighbours': [
            miro_id_to_identifiers(miro_id)
            for miro_id in neighbour_ids
        ]
    }


@app.get('/healthcheck')
def healthcheck():
    return {'status': 'healthy'}
