import json
from datetime import datetime
from io import BytesIO

import httpx
from PIL import Image
from skimage import color
from sklearn.cluster import KMeans
from src.elasticsearch import get_elastic_client
from src.log import get_logger
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

log = get_logger()
es = get_elastic_client()

# create an index with the mapping in /data/index_config/images.json
with open("data/index_config/images.json", "r", encoding="utf-8") as f:
    index_config = json.load(f)

todays_date_iso = datetime.today().strftime("%Y-%m-%d")
index_name = f"images-color-knn-{todays_date_iso}"
if es.indices.exists(index=index_name):
    log.info(f"Deleting index {index_name}")
    es.indices.delete(index=index_name)
log.info(f"Creating index {index_name}")
es.indices.create(index=index_name, **index_config)


base_url = "https://api.wellcomecollection.org/catalogue/v2/images"
page = 1
response = httpx.get(base_url, params={"pageSize": "100", "page": page}).json()
progress_bar = tqdm(total=response["totalResults"])
while page <= response["totalPages"]:
    for image_data in response["results"]:
        log.debug(
            f"Processing image {image_data['id']} "
            f"{progress_bar.n} / {progress_bar.total}"
        )
        progress_bar.set_description(f'Processing {image_data["id"]}')

        # get the image
        iiif_url = image_data["thumbnail"]["url"]
        thumbnail_url = iiif_url.replace(
            "info.json",
            "full/!200,200/0/default.jpg",
        )
        thumbnail = httpx.get(thumbnail_url).content
        rgb_image = Image.open(BytesIO(thumbnail))

        # transform to lab space

        lab_image = color.rgb2lab(rgb_image)

        # cluster pixels to get 5 dominant colours
        pixels = lab_image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(pixels)
        dominant_colours = kmeans.cluster_centers_

        # index them as little 3d lab arrays
        document = {
            "thumbnail_url": thumbnail_url,
            "title": image_data["source"]["title"],
            "colors": {
                "a": dominant_colours[0].tolist(),
                "b": dominant_colours[1].tolist(),
                "c": dominant_colours[2].tolist(),
                "d": dominant_colours[3].tolist(),
                "e": dominant_colours[4].tolist(),
            },
        }
        es.index(
            index=index_name,
            document=document,
            id=image_data["id"],
        )
        progress_bar.update(1)

    page += 1
    log.debug(f"Getting page {page} of {response['totalPages']}")
    progress_bar.set_description(f"Getting page {page} of {response['totalPages']}")
    response = httpx.get(
        base_url, params={"pageSize": "100", "page": page}
    ).json()
