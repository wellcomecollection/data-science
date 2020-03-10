from fastapi import FastAPI, HTTPException

from src.image import get_image_from_url, get_image_from_iiif_url
from src.lsh import LSHEncoder
from src.feature_extraction import extract_features
from src.logging import logger

lsh_encoder = LSHEncoder('2020-03-05')

# initialise API
app = FastAPI(
    title='Feature Vector',
    description='Takes an image url and returns the image\'s feature vector encoded as an LSH string',
    docs_url='/feature-vector/docs',
    redoc_url='/feature-vector/redoc'
)


@app.get('/feature-vector/')
def feature_similarity_by_catalogue_id(image_url: str = None, iiif_url: str = None):
    if image_url:
        image = get_image_from_url(image_url)
        url = image_url
    elif iiif_url:
        image = get_image_from_iiif_url(iiif_url)
        url = iiif_url
    else:
        logger.error('No URL provided')
        raise HTTPException(status_code=400, detail='No URL provided')

    features = extract_features(image)
    lsh_encoded_features = lsh_encoder(features)

    # elasticsearch can only handle 2048-dimensional dense vectors, so we're
    # splitting them in the response.
    feature_vector_1, feature_vector_2 = features.reshape(2, 2048).tolist()

    logger.info(f'extracted features from image at: {url}')

    return {
        'feature_vector_1': feature_vector_1,
        'feature_vector_2': feature_vector_2,
        'lsh_encoded_features': lsh_encoded_features
    }


@app.get('/healthcheck')
def healthcheck():
    return {'status': 'healthy'}
