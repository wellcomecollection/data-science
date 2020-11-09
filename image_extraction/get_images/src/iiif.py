from weco_datascience.http import fetch_url_json
from weco_datascience.logging import get_logger

log = get_logger(__name__)


async def get_images_from_iiif_manifest(iiif_manifest_url):
    manifest = await fetch_url_json(iiif_manifest_url)

    image_urls = [
        image["resource"]["@id"]
        for sequence in manifest["json"]["sequences"]
        for canvas in sequence["canvases"]
        for image in canvas["images"]
    ]
    log.info(f"Found {len(image_urls)} image urls at {iiif_manifest_url}")
    return image_urls
