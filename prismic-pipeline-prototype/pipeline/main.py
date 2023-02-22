import json
from datetime import datetime
from pathlib import Path

from src.elasticsearch import get_elastic_client
from src.log import get_logger
from src.prismic import yield_events, yield_exhibitions, yield_stories
from src.transform import transform_data

log = get_logger()

# make the data directories
data_dir = Path("/data/prismic")
types = [
    "stories",
    "exhibitions",
    "events"
]
for type in types:
    (data_dir / type).mkdir(parents=True, exist_ok=True)

# if it hasn't already been done, save all of the data from prismic locally
# first check whether there are any files in the data directories
if any((data_dir / type).iterdir()):
    log.info("Data already exists locally")
else:
    log.info("Saving all of the stories locally")
    for story in yield_stories(batch_size=100, limit=None):
        log.info(f"Saved story {story['id']}")
        Path(data_dir / "stories" / f"{story['id']}.json").write_text(
            json.dumps(story), encoding="utf-8"
        )

    log.info("Saving all of the exhibitions locally")
    for exhibition in yield_exhibitions(batch_size=100, limit=None):
        Path(data_dir / "exhibitions" / f"{exhibition['id']}.json").write_text(
            json.dumps(exhibition), encoding="utf-8"
        )
        log.info(f"Saved exhibition {exhibition['id']}")

    log.info("Saving all of the events locally")
    for event in yield_events(batch_size=100, limit=None):
        Path(data_dir / "events" / f"{event['id']}.json").write_text(
            json.dumps(event), encoding="utf-8"
        )
        log.info(f"Saved event {event['id']}")


# format the data according to the index config and post it to elasticsearch
todays_date = datetime.now().strftime("%Y-%m-%d")
es = get_elastic_client()
for type in types:
    index_name = f"prismic-{type}-{todays_date}"

    index_config = json.loads(
        Path(f"data/index_config/{type}.json").read_text()
    )

    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    es.indices.create(
        index=index_name,
        mappings=index_config["mappings"],
        settings=index_config["settings"],
    )

    for file in (data_dir / type).iterdir():
        raw_data = json.loads(file.read_text(encoding="utf-8"))
        id, document = transform_data(raw_data, type=type)
        log.info(f"Indexing {type} {id}")
        es.index(
            index=f"prismic-{type}-{todays_date}", id=id, document=document
        )
