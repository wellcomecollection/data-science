import json
import os
from pathlib import Path

from src.elasticsearch import (
    get_catalogue_elastic_client,
    get_concepts_elastic_client,
    get_local_elastic_client,
)
from src.log import get_logger
from src.sources import (
    get_total_stories,
    yield_concepts,
    yield_stories,
    yield_works,
)
from src.wellcome import works_index
from tqdm import tqdm

log = get_logger()

max_docs = 1000
local_index_name = os.environ.get("LOCAL_INDEX_NAME")
config_path = Path("/data/index_config").absolute()
with open(config_path / "search-as-you-type.json", "r") as f:
    config = json.load(f)


log.info("Loading local elastic client")
local_es = get_local_elastic_client()

log.info("Creating index")
local_es.indices.delete(index=local_index_name, ignore_unavailable=True)
local_es.indices.create(index=local_index_name, **config)


# WORKS
log.info("Loading catalogue elastic client")
catalogue_es = get_catalogue_elastic_client()
total_works = catalogue_es.count(index=works_index)["count"]

log.info("Indexing works")
works_generator = yield_works(
    es=catalogue_es, index=works_index, batch_size=100, limit=max_docs
)

for result in tqdm(works_generator, total=min(total_works, max_docs)):
    display_data = result["_source"]["display"]
    description = ""
    if "description" in display_data:
        description = display_data["description"]
    local_es.index(
        index=local_index_name,
        id=result["_id"],
        document={
            "type": "work",
            "title": display_data["title"],
            "description": description,
        },
    )


# CONCEPTS
log.info("Loading concepts elastic client")
concepts_index = "concepts-store"
concepts_es = get_concepts_elastic_client()
total_concepts = concepts_es.count(index=concepts_index)["count"]

log.info("Indexing concepts")
concepts_generator = yield_concepts(
    es=concepts_es, index=concepts_index, batch_size=100, limit=max_docs
)

for result in tqdm(concepts_generator, total=min(total_concepts, max_docs)):
    local_es.index(
        index=local_index_name,
        id=result["_id"],
        document={
            "type": "concept",
            "title": result["_source"]["query"]["label"],
        },
    )


# STORIES
total_stories = get_total_stories()

log.info("Indexing stories")
stories_generator = yield_stories(batch_size=100, limit=max_docs)
for result in tqdm(stories_generator, total=min(total_stories, max_docs)):
    standfirst = ""
    for item in result["data"]["body"]:
        if item["slice_type"] == "standfirst":
            standfirst = item["primary"]["text"][0]["text"]
            break

    local_es.index(
        index=local_index_name,
        id=result["id"],
        document={
            "type": "story",
            "title": result["data"]["title"][0]["text"],
            "description": standfirst,
        },
    )


# count the docs in the index
log.info("Counting docs in index")
count = local_es.count(index=local_index_name)["count"]
log.info(f"Indexed {count} docs")
