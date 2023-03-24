import os
import json


from tqdm import tqdm

from src.elasticsearch import get_elastic_client
from src.log import get_logger
from src.prismic import count_documents, yield_documents
from src.embed import TextEmbedder

log = get_logger()

log.info("Connecting to target elasticsearch client")
target_es = get_elastic_client()
target_index = f"prismic-{os.environ['MODEL_NAME']}"

log.info(f"Creating target index: {target_index}")
with open("/data/index_config/documents.json", encoding="utf-8") as f:
    index_config = json.load(f)

if target_es.indices.exists(index=target_index):
    target_es.indices.delete(index=target_index)
target_es.indices.create(index=target_index, **index_config)

log.info("Loading sentence transformer model")
model = TextEmbedder(
    model=os.environ["MODEL_NAME"], cache_dir="/data/embeddings"
)


progress_bar = tqdm(yield_documents(batch_size=100), total=count_documents())
for document in progress_bar:
    actions = []
    for i, slice in enumerate(document["data"]["body"]):
        if slice["slice_type"] in ["text", "standfirst", "quoteV2"]:
            title = document["data"]["title"][0]["text"]
            text = "\n".join(
                [paragraph["text"] for paragraph in slice["primary"]["text"]]
            ).strip()
            if not text:
                continue
            actions.append(
                {"index": {"_index": target_index, "_id": f"{document['id']}-slice-{i}"}})
            actions.append({
                "id": document["id"],
                "title": title,
                "title-embedding": model.embed(title),
                "text": text,
                "text-embedding": model.embed(text),
            })

    if actions:
        target_es.bulk(operations=actions)
        progress_bar.set_description(f"Indexed {document['id']}")
