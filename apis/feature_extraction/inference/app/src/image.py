from io import BytesIO
from urllib.parse import unquote_plus

import requests
from PIL import Image

from .logging import logger


def is_valid_image_url(image_url):
    image_formats = ['image/png', 'image/jpeg', 'image/jpg']
    try:
        r = requests.head(image_url)
        if (r.status_code == 200) and (r.headers['content-type'] in image_formats):
            return True
        return False
    except:
        return False


def get_image_from_url(image_url):
    image_url = unquote_plus(image_url)
    if is_valid_image_url(image_url):
        r = requests.get(image_url)
        image = Image.open(BytesIO(r.content))
        return image
    else:
        message = f'{image_url} is not a valid image URL'
        logger.error(message)
        raise ValueError(message)


def get_image_from_iiif_url(iiif_url):
    url = unquote_plus(iiif_url)
    image_url = url.replace('info.json', '/full/760,/0/default.jpg')
    image = get_image_from_url(image_url)
    return image
