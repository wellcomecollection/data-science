import json
import os
from pathlib import Path

from tqdm import tqdm

from src.elasticsearch import (
    get_catalogue_elastic_client,
    get_concepts_elastic_client,
    get_concepts_prototype_elastic_client,
    get_local_elastic_client,
)
from src.log import get_logger
from src.sources import (
    count_stories,
    count_works,
    get_contributors,
    yield_concepts,
    yield_stories,
    yield_works,
)
from src.wellcome import works_index

log = get_logger()

max_docs = None
target_index = os.environ.get("INDEX_NAME")
config_path = Path("/data/index_config").absolute()
with open(config_path / "search-as-you-type.json", "r") as f:
    config = json.load(f)


log.info("Loading target elastic client")
target_es = get_concepts_prototype_elastic_client()

for index in target_es.indices.get_alias(index="search-as-you-type*").keys():
    log.info(f"Deleting index {index}")
    target_es.indices.delete(index=index)

log.info("Creating target index")
target_es.indices.create(index=target_index, **config)


# WORKS
log.info("Loading catalogue elastic client")
catalogue_es = get_catalogue_elastic_client()

log.info("Indexing works")
total_works = count_works(catalogue_es, works_index)
works_generator = yield_works(
    es=catalogue_es, index=works_index, batch_size=100
)
progress_bar = tqdm(
    total=(min(total_works, max_docs) if max_docs else total_works)
)
for batch in works_generator:
    operations_batch = []
    for hit in batch:
        display_data = hit["_source"]["display"]
        contributors = ", ".join(
            [
                contributor["agent"]["label"]
                for contributor in display_data.get("contributors", [])
            ]
        )
        thumbnail_url = None
        if "thumbnail" in display_data:
            thumbnail_url = display_data["thumbnail"]["url"]

        operations_batch.extend(
            [
                {"index": {"_index": target_index, "_id": hit["_id"]}},
                {
                    "type": "work",
                    "title": display_data["title"],
                    "contributors": contributors,
                    "url": f"https://wellcomecollection.org/works/{hit['_id']}",
                    "image": thumbnail_url,
                },
            ]
        )
    target_es.bulk(index=target_index, operations=operations_batch)
    progress_bar.update(len(batch))

# CONCEPTS
log.info("Loading concepts elastic client")
concepts_index = "concepts-store"
concepts_es = get_concepts_elastic_client()

log.info("Indexing concepts")
total_concepts = concepts_es.count(index=concepts_index)["count"]
concepts_generator = yield_concepts(
    es=concepts_es, index=concepts_index, batch_size=100
)
progress_bar = tqdm(
    total=(min(total_concepts, max_docs) if max_docs else total_concepts),
)
for batch in concepts_generator:
    operations_batch = []
    for hit in batch:
        operations_batch.extend(
            [
                {"index": {"_index": target_index, "_id": hit["_id"]}},
                {
                    "type": "concept",
                    "title": hit["_source"]["query"]["label"],
                    "url": f"https://wellcomecollection.org/concepts/{hit['_id']}",
                },
            ]
        )
    target_es.bulk(index=target_index, operations=operations_batch)
    progress_bar.update(len(batch))


# STORIES
log.info("Indexing stories")
total_stories = count_stories()
stories_generator = yield_stories(batch_size=100, limit=max_docs)
for result in tqdm(
    stories_generator,
    total=(min(total_stories, max_docs) if max_docs else total_stories),
):
    contributors = get_contributors(result)
    try:
        thumbnail_url = result["data"]["promo"][0]["primary"]["image"]["url"]
    except (KeyError, IndexError):
        thumbnail_url = None
    target_es.index(
        index=target_index,
        id=result["id"],
        document={
            "type": "story",
            "title": result["data"]["title"][0]["text"],
            "contributors": ", ".join(contributors),
            "url": f"https://wellcomecollection.org/articles/{result['id']}",
            "image": thumbnail_url,
        },
    )


# count the docs in the index
log.info("Counting docs in index")
count = target_es.count(index=target_index)["count"]
log.info(f"Indexed {count} docs")
