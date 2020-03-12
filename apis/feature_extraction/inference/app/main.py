from fastapi import FastAPI, HTTPException

from src.image import get_image_from_url, get_image_url_from_iiif_url
from src.lsh import LSHEncoder
from src.feature_extraction import extract_features

lsh_encoder = LSHEncoder()

# initialise API
app = FastAPI(
    title='Feature Vector',
    description='Takes an image url and returns the image\'s feature vector encoded as an LSH string',
    docs_url='/feature-vector/docs',
    redoc_url='/feature-vector/redoc'
)


@app.get('/feature-vector/')
def feature_similarity_by_catalogue_id(image_url: str = None, iiif_url: str = None):
    if (not (image_url or iiif_url)) or (iiif_url and image_url):
        raise HTTPException(
            status_code=400,
            detail='API takes one of: image_url, iiif_url'
        )

    if iiif_url:
        try:
            image_url = get_image_url_from_iiif_url(iiif_url)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail='iiif_url is not a valid iiif url'
            )

    try:
        image = get_image_from_url(image_url)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f'{image_url} is not a valid image url'
        )

    features = extract_features(image)
    lsh_encoded_features = lsh_encoder(features)

    return {
        'features': features,
        'lsh_encoded_features': lsh_encoded_features
    }


@app.get('/healthcheck')
def healthcheck():
    return {'status': 'healthy'}
