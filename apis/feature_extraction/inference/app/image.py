from io import BytesIO

import requests
from PIL import Image


def get_image_from_url(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    return image
