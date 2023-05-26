import json
import os
from datetime import datetime
from io import BytesIO

import httpx
from PIL import Image
from src.elasticsearch import get_prototype_elastic_client
from src.source import yield_source_images, count_source_images
from src.embedder import ColorEmbedder
from src.log import get_logger
from tqdm import tqdm


log = get_logger()
es = get_prototype_elastic_client()

n_bins = int(os.environ.get("N_BINS", 6))

with open("data/index_config/images.json", "r", encoding="utf-8") as f:
    index_config = json.load(f)

index_config = json.loads(
    json.dumps(index_config).replace(
        "{{EMBEDDING_DIMENSIONALITY}}", str(n_bins ** 3)
    )
)

todays_date_iso = datetime.today().strftime("%Y-%m-%d")
index_name = f"images-color-embedding-{todays_date_iso}"
if es.indices.exists(index=index_name):
    log.info(f"Deleting index {index_name}")
    es.indices.delete(index=index_name)
log.info(f"Creating index {index_name}")
es.indices.create(index=index_name, **index_config)

color_embedder = ColorEmbedder(n_bins=n_bins)



progress_bar = tqdm(
    yield_source_images(pipeline_date="2023-03-29"), 
    total=count_source_images(pipeline_date="2023-03-29")
)

for image_data in progress_bar:
    log.debug(
        f"Processing image {image_data['id']}"
        f"{progress_bar.n} / {progress_bar.total}"
    )
    progress_bar.set_description(f'Processing {image_data["id"]}')

    # get the image
    iiif_url = image_data["thumbnail"]["url"]
    thumbnail_url = iiif_url.replace(
        "info.json",
        "full/!400,400/0/default.jpg",
    )
    thumbnail = httpx.get(thumbnail_url).content
    image = Image.open(BytesIO(thumbnail))
    embedding = color_embedder.embed(image)

    document = {
        "thumbnail_url": thumbnail_url,
        "image_id": image_data["id"],
        "source_id": image_data["source"]["id"],
        "title": image_data["source"]["title"],
        "embedding": embedding.tolist(),
    }

    es.index(
        index=index_name,
        document=document,
        id=image_data["id"],
    )
