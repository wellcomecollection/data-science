import requests
from io import BytesIO
from os.path import basename, splitext

from PIL import Image

from .aws import get_object_from_s3


def get_image(s3_bucket_name, s3_object_key):
    try:
        image_bytes = get_object_from_s3(
            bucket_name=s3_bucket_name,
            object_key=s3_object_key,
            profile_name='platform-dev'
        )
        image = Image.open(BytesIO(image_bytes))
    except Image.DecompressionBombError:
        miro_id = splitext(basename(s3_object_key))[0]
        image_url = f'https://iiif.wellcomecollection.org/image/{miro_id}.jpg/full/700,/0/default.jpg'
        image_bytes = requests.get(image_url).content
        image = Image.open(BytesIO(image_bytes))
    return image
