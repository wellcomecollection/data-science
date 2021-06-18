import httpx
from PIL import Image
from io import BytesIO
from piffle.iiif import IIIFImageClient, ParseError


def get_image_url_from_iiif_url(iiif_url, input_size=224):
    try:
        image = IIIFImageClient.init_from_url(iiif_url)
    except ParseError:
        raise ValueError(f"{iiif_url} is not a valid iiif URL")

    if "dlcs" in image.api_endpoint:
        image.api_endpoint = image.api_endpoint.replace("iiif-img", "thumbs")
        return str(image.size(width=400, height=400, exact=True))

    return str(image.size(width=input_size, height=input_size, exact=False))


def get_image_from_url(image_url):
    response = httpx.get(image_url)
    image = Image.open(BytesIO(response.content))
    return image
