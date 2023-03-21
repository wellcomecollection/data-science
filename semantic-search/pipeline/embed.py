import json
from datetime import datetime

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.elasticsearch import get_elastic_client
from src.log import get_logger
from src.prismic import count_articles, yield_articles

log = get_logger()

log.info("Connecting to target elasticsearch client")
target_es = get_elastic_client()
todays_date = datetime.now().strftime("%Y-%m-%d")
target_index = f"enriched-articles-{todays_date}"

log.info("Creating target index")
with open("/data/index_config/documents.json", encoding="utf-8") as f:
    index_config = json.load(f)

if target_es.indices.exists(index=target_index):
    target_es.indices.delete(index=target_index)
target_es.indices.create(index=target_index, **index_config)

log.info("Loading sentence transformer model")
model = SentenceTransformer(
    model_name_or_path="paraphrase-distilroberta-base-v1",
    cache_folder="/data/models",
)


progress_bar = tqdm(yield_articles(batch_size=100), total=count_articles())
for article in progress_bar:
    for i, slice in enumerate(article["data"]["body"]):
        if slice["slice_type"] in ["text", "standfirst", "quoteV2"]:
            for j, paragraph in enumerate(slice["primary"]["text"]):
                text = paragraph["text"]
                embedding = model.encode(text)
                document_id = f"{article['id']}-slice-{i}-paragraph-{j}"
                target_es.index(
                    index=target_index,
                    id=document_id,
                    document={
                        "id": article["id"],
                        "text": text,
                        "embedding": embedding,
                    },
                )
                progress_bar.set_description(f"Embedding {document_id}")
