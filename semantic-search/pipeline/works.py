import os
import json


from tqdm import tqdm

from src.elasticsearch import get_elastic_client
from src.log import get_logger
from src.wellcome import count_works, yield_popular_works
from src.embed import TextEmbedder

log = get_logger()

log.info("Connecting to target elasticsearch client")
target_es = get_elastic_client()
target_index = f"works-{os.environ['MODEL_NAME']}"

log.info(f"Creating target index: {target_index}")
with open("/data/index_config/documents.json", encoding="utf-8") as f:
    index_config = json.load(f)

if target_es.indices.exists(index=target_index):
    target_es.indices.delete(index=target_index)
target_es.indices.create(index=target_index, **index_config)

log.info("Loading sentence transformer model")
model = TextEmbedder(
    model=os.environ["MODEL_NAME"], cache_dir="/data/embeddings")


i = 0
operations = []
batch_start_id = None
n_works_to_index = 5000
progress_bar = tqdm(
    yield_popular_works(size=n_works_to_index),
    total=n_works_to_index
)
for work in progress_bar:
    i += 1
    progress_bar.set_description(f"Embedding {work['id']}")
    log.debug(f"Embedding {work['id']} title")

    operations.append(
        {"index": {"_index": target_index, "_id": f"{work['id']}-title"}}
    )
    operations.append(
        {
            "id": work["id"],
            "type": work["workType"]["label"],
            "title": work["title"],
            "text": None,
            "embedding": model.embed(work["title"]),
        }
    )

    if "description" in work:
        log.debug(f"Embedding {work['id']} description")
        operations.append(
            {"index": {"_index": target_index,
                       "_id": f"{work['id']}-description"}}
        )
        operations.append(
            {
                "id": work["id"],
                "type": "works",
                "format": work["workType"]["label"],
                "title": work["title"],
                "text": work["description"],
                "embedding": model.embed(work["description"]),
            }
        )

    if i % 100 == 0:
        target_es.bulk(operations=operations)
        progress_bar.set_description(
            f"Indexing batch {batch_start_id}-{work['id']}"
        )
        log.debug(f"Indexing batch {batch_start_id}-{work['id']}")
        operations = []
        batch_start_id = work["id"]
