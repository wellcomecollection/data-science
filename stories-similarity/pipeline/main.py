import numpy as np
import json
import os

from sentence_transformers import SentenceTransformer
from src.elasticsearch import get_elastic_client
from src.log import get_logger
from src.stories import count_stories, yield_stories
from tqdm import tqdm

log = get_logger()

log.info("Connecting to target elasticsearch client")
target_es = get_elastic_client()
target_index = os.environ.get("ES_INDEX")

log.info("Creating target index")
with open("/data/index_config/enriched-stories.json", encoding="utf-8") as f:
    index_config = json.load(f)
if target_es.indices.exists(target_index):
    target_es.indices.delete(index=target_index)
target_es.indices.create(index=target_index, **index_config)

log.info("Loading sentence transformer model")
model = SentenceTransformer(
    model_name_or_path="paraphrase-distilroberta-base-v1",
    cache_folder="/data/models",
)

for result in tqdm(yield_stories(batch_size=100), total=count_stories()):
    title = result["data"]["title"][0]["text"]
    title_embedding = model.encode(title)

    standfirst = ""
    for item in result["data"]["body"]:
        if item["slice_type"] == "standfirst":
            standfirst = item["primary"]["text"][0]["text"]
            break
    standfirst_embedding = model.encode(standfirst)

    try:
        thumbnail_url = result["data"]["promo"][0]["primary"]["image"]["url"]
    except (KeyError, IndexError):
        thumbnail_url = None

    shared_embedding_max = np.maximum(title_embedding, standfirst_embedding)
    shared_embedding_concat = np.concatenate(
        [title_embedding[:512], standfirst_embedding[:512]]
    )
    target_es.index(
        index=target_index,
        id=result["id"],
        document={
            "url": f"https://wellcomecollection.org/articles/{result['id']}",
            "thumbnail": thumbnail_url,
            "title": title,
            "title_embedding": title_embedding,
            "standfirst": standfirst,
            "standfirst_embedding": standfirst_embedding,
            "shared_embedding_max": shared_embedding_max,
            "shared_embedding_concat": shared_embedding_concat
        }
    )
