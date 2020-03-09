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

# initialise API
app = FastAPI(
    title='Feature Vector',
    description='Takes an image url and returns the image\'s feature vector encoded as an LSH string',
    docs_url='/feature-vector/docs',
    redoc_url='/feature-vector/redoc'
)


@app.get('/feature-vector/')
def feature_similarity_by_catalogue_id(image_url: str):
    image = get_image_from_url(unquote_plus(image_url))
    features = extract_features(image)
    lsh_encoded_features = lsh_encoder(features)
    feature_vector_1, feature_vector_2 = features.reshape(2, 2048).tolist()

    return {
        'feature_vector_1': feature_vector_1,
        'feature_vector_2': feature_vector_2,
        'lsh_encoded_features': lsh_encoded_features
    }


@app.get('/healthcheck')
def healthcheck():
    return {'status': 'healthy'}
