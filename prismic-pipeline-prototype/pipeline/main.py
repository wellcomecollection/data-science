import json
from datetime import datetime
from pathlib import Path

from src.elasticsearch import get_elastic_client
from src.log import get_logger
from src.prismic import yield_documents
from src.transform import transform_data

log = get_logger()

# make the data directories
data_dir = Path("/data/prismic")
data_dir.mkdir(parents=True, exist_ok=True)

types_to_fetch = [
    "articles",
    "exhibitions",
    "events",
    "people",
    "series",
    "webcomic-series",
    "event-series"
]
types_to_index = [
    "articles",
    "exhibitions",
    "events",
]
for document_type in types_to_fetch:
    type_data_dir = data_dir / document_type
    type_data_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving all of the {document_type} locally")
    for document in yield_documents(batch_size=100, limit=None, document_type=document_type):
        Path(type_data_dir / f"{document['id']}.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
        log.info(f"Saved {document_type} {document['id']}")

# format the data according to the index config and post it to elasticsearch
todays_date = datetime.now().strftime("%Y-%m-%d")
es = get_elastic_client()
for document_type in types_to_index:
    index_name = f"prismic-{document_type}-{todays_date}"
    index_config = json.loads(
        Path(
            f"data/index_config/{document_type}.json"
        ).read_text(encoding="utf-8")
    )

    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    es.indices.create(
        index=index_name,
        mappings=index_config["mappings"],
        settings=index_config["settings"],
    )

    for file in (data_dir / document_type).iterdir():
        raw_data = json.loads(file.read_text(encoding="utf-8"))
        id, document = transform_data(raw_data, document_type=document_type)
        log.info(f"Indexing {document_type} {id}")
        es.index(
            index=f"prismic-{document_type}-{todays_date}",
            id=id,
            document=document
        )
