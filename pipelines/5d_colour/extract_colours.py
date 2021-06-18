import json
from src.image import get_image_from_url

with open("/data/image_urls.json", 'r') as f:
    image_urls = json.load(f)

image = get_image_from_url(image_urls[0])

