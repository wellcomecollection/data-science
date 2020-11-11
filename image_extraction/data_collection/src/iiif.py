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


async def get_ocr_images_from_manifest(iiif_manifest_url):
    manifest = await fetch_url_json(iiif_manifest_url)
    annotation_list = await fetch_url_json(
        str(iiif_manifest_url.replace("manifest", "images"))
    )

    annotations = {}
    for annotation in annotation_list["json"]["resources"]:
        url, xywh = annotation["on"].split("#xywh=")
        annotations[url] = xywh

    image_urls = []
    for sequence in manifest["json"]["sequences"]:
        for canvas in sequence["canvases"]:
            if canvas["@id"] in annotations:
                xywh = annotations[canvas["@id"]]
                image_url = canvas["images"][0]["resource"]["service"]["@id"]
                crop_url = f"{image_url}/{xywh}/full/0/default.jpg"
                image_urls.append(crop_url)

    log.info(f"Found {len(image_urls)} image urls at {iiif_manifest_url}")
    return image_urls
