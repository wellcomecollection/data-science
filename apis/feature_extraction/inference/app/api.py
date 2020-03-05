import json
import re
from urllib.parse import unquote_plus
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from .image import get_image_from_url
from .lsh import LSHEncoder
from .feature_extraction import extract_features

lsh_encoder = LSHEncoder('2020-03-05')


class Image(BaseModel):
    url: str


# initialise API
app = FastAPI(
    title='Feature Vector',
    description='Takes an image url and returns the image\'s feature vector encoded as an LSH string',
    docs_url='/feature-vector/docs',
    redoc_url='/feature-vector/redoc'
)


@app.get('/feature-vector/{image_id}')
def feature_similarity_by_catalogue_id(image_id: str):
    image_url = f'https://iiif.wellcomecollection.org/image/{image_id}.jpg/full/960,/0/default.jpg'
    image = get_image_from_url(image_url)
    feature_vector = extract_features(image)
    lsh_encoded_features = lsh_encoder(feature_vector)
    return {
        'feature_vector': feature_vector.tolist(),
        'lsh_encoded_features': lsh_encoded_features
    }


@app.get('/healthcheck')
def healthcheck():
    return {'status': 'healthy'}
