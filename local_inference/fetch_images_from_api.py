#!/usr/bin/env python3

import asyncio
import click
import math
import os
from tqdm import tqdm
from weco_datascience import http
from weco_datascience.image import get_image_from_url

CATALOGUE_API = "https://api-stage.wellcomecollection.org/catalogue/v2/images"
MAX_PAGESIZE = 100


async def get_iiif_urls_from_page(page, page_size):
    url = f"{CATALOGUE_API}?pageSize={page_size}&page={page}"
    data = await http.fetch_url_json(url)
    try:
        return [
            (image["id"], image["locations"][0]["url"])
            for image in data["json"]["results"]
        ]
    except KeyError:
        print(data)


async def save_image_from_url(url, size, id, dir):
    image = await get_image_from_url(url, size)
    path = os.path.join(dir, f"{id}.jpg")
    image.save(path)


async def download_images_from_page(page, page_size, image_size, dir):
    iiif_urls = await get_iiif_urls_from_page(page, page_size)
    save_requests = [
        save_image_from_url(url, image_size, id, dir)
        for id, url in iiif_urls
    ]
    await asyncio.gather(*save_requests)
    return len(iiif_urls)


@click.command()
@click.option("-n", "--n-images", type=int, required=True)
@click.option("-o", "--output-dir", type=str, required=True)
@click.option("--image-size", type=int, default=224)
@click.option("--page-offset", type=int, default=0)
def fetch_images(n_images, output_dir, image_size, page_offset):
    n_pages = math.ceil(n_images / MAX_PAGESIZE)

    def get_page_size(page):
        if page != n_pages - 1:
            return MAX_PAGESIZE
        else:
            mod = n_images % MAX_PAGESIZE
            return mod if mod != 0 else MAX_PAGESIZE

    print(f"Downloading {n_images} images from {n_pages} pages...")
    http.start_persistent_client_session()

    async def run():
        # Why not do these concurrently, you ask?
        # Because it kills the API :(
        for current_page in tqdm(range(n_pages)):
            await download_images_from_page(
                page=(current_page + page_offset + 1),
                page_size=get_page_size(current_page + page_offset),
                image_size=image_size,
                dir=output_dir
            )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
    loop.run_until_complete(http.close_persistent_client_session())


if __name__ == "__main__":
    fetch_images()
