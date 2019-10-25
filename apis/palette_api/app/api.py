import itertools
import json
import os
import pathlib
import pickle
import re
from io import BytesIO

import nmslib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from .aws import get_object_from_s3
from .colours import hex_to_rgb, random_hex, rgb_to_lab
from .identifiers import (catalogue_id_to_miro_id, catalogue_ids,
                          miro_id_to_identifiers, miro_ids,
                          miro_ids_cleared_for_catalogue_api)
from .neighbours import get_neighbour_ids, palette_index
from .palette_embedder import embed_hex_palette

# initialise API
app = FastAPI(
    title='Palette Similarity',
    description='Find similar images based on their colour, using approximate embeddings of euclidean distance in LAB space between 5-colour palettes',
    docs_url='/docs',
    redoc_url='/redoc'
)

# create API endpoints


@app.get('/works/{catalogue_id}')
def palette_similarity_by_catalogue_id(catalogue_id: str, n: int = 10):
    catalogue_id = catalogue_id or np.random.choice(catalogue_ids)
    if catalogue_id not in catalogue_ids:
        raise HTTPException(status_code=404, detail="Invalid catalogue id")

    miro_id = catalogue_id_to_miro_id[catalogue_id]
    query_index = np.where(miro_ids == miro_id)[0][0]
    query_embedding = np.array(palette_index[query_index]).reshape(1, -1)
    neighbour_ids = get_neighbour_ids(
        query_embedding, n, skip_first_result=True)
    print(neighbour_ids)

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
