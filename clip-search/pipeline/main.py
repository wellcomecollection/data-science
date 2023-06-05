import os
import json
from datetime import datetime
from io import BytesIO

import torch
import clip

import httpx
from PIL import Image
from src.elasticsearch import get_prototype_elastic_client
from src.source import yield_source_images, count_source_images
from src.log import get_logger
from tqdm import tqdm


log = get_logger()
es = get_prototype_elastic_client()


with open("data/index_config/images.json", "r", encoding="utf-8") as f:
    index_config = json.load(f)

todays_date_iso = datetime.today().strftime("%Y-%m-%d")
index_name = f"images-clip-{todays_date_iso}"
if es.indices.exists(index=index_name):
    log.info(f"Deleting index {index_name}")
    es.indices.delete(index=index_name)
log.info(f"Creating index {index_name}")
es.indices.create(index=index_name, **index_config)

log.info("Loading CLIP model")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = os.environ["MODEL_NAME"]
model, preprocessor = clip.load(
    model_name, download_root="/data/models", device=device
)

progress_bar = tqdm(
    yield_source_images(pipeline_date="2023-03-29"),
    total=count_source_images(pipeline_date="2023-03-29"),
)

for image_data in progress_bar:
    try:
        log.debug(
            f"Processing image {image_data['id']} "
            f"{progress_bar.n}/{progress_bar.total}"
        )
        progress_bar.set_description(f'Processing {image_data["id"]}')

        # get the image
        iiif_url = image_data["thumbnail"]["url"]
        thumbnail_url = iiif_url.replace(
            "info.json",
            "full/!400,400/0/default.jpg",
        )
        thumbnail = httpx.get(thumbnail_url, timeout=30).content
        image = Image.open(BytesIO(thumbnail))

        # get the embedding
        image_input = preprocessor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input).squeeze(0)

        # make sure the embedding is of unit length so that we can use the
        # dot product similarity
        embedding /= embedding.norm(dim=-1, keepdim=True)

        # index the image
        document = {
            "thumbnail_url": thumbnail_url,
            "image_id": image_data["id"],
            "source_id": image_data["source"]["id"],
            "title": image_data["source"]["title"],
            "embedding": embedding.tolist(),
        }

        es.index(index=index_name, document=document, id=image_data["id"])
    except Exception as e:
        log.error(f"Error processing image {image_data['id']}: {e}")

    progress_bar.update(1)
