import json
import re
from urllib.parse import quote, unquote
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from .image import get_image_from_url
from .lsh import LSHEncoder

lsh_encoder = LSHEncoder('2020-03-05')

class Image(BaseModel):
    url: str

# initialise API
app = FastAPI(
    title='Feature Vector',
    description='Takes an image url, and returns the image\'s feature vector encoded as an LSH string',
    docs_url='/feature-vector/docs',
    redoc_url='/feature-vector/redoc'
)


@app.get('/feature-vector/{image_url}')
def feature_similarity_by_catalogue_id(image: Image):
    image = get_image_from_url(image['url'])
    image.
    feature_vector = ''
    return {
        'feature_vector': feature_vector
    }


@app.get('/healthcheck')
def healthcheck():
    return {'status': 'healthy'}
