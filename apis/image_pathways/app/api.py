import os
import string
from random import getrandbits

import numpy as np
from fastapi import FastAPI, HTTPException

from .utils import get_pathway, get_random_image_id, id_to_url, ids

app = FastAPI(
    title='Image Pathways',
    description='Find visual pathways between images in Wellcome Collection',
    docs_url='/image_pathways/docs',
    redoc_url='/image_pathways/redoc'
)


@app.get('/image_pathways')
def pathway(id_1: str = None, id_2: str = None, path_length: int = 10):
    id_1 = id_1 or get_random_image_id()
    id_2 = id_2 or get_random_image_id(excluding=id_1)

    if ((id_1 not in ids) or (id_2 not in ids)):
        raise HTTPException(status_code=404, detail="Invalid image_id")

    id_path = get_pathway(id_1, id_2, path_length)

    image_url_path = [id_to_url(image_id) for image_id in id_path]

    return {
        'id_path': id_path,
        'image_url_path': image_url_path
    }


@app.get('/image_pathways/health_check')
def health_check():
    return {'status': 'healthy'}
