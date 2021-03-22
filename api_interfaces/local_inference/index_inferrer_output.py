#!/usr/bin/env python3

import click
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
import json
from tqdm import tqdm
import re

from palette_adapter import PaletteAdapter

ADAPTERS = {
    "palette": PaletteAdapter
}


@click.command()
@click.option("-e", "--elasticsearch-connection-string", type=str, required=False)
@click.option("-f", "--input-file", type=str, required=True)
@click.option("-i", "--index", type=str, required=True)
@click.option("-a", "--adapter-name", type=click.Choice(list(ADAPTERS.keys())), required=True)
def index_inferrer_output(elasticsearch_connection_string, input_file, index, adapter_name):
    es = Elasticsearch(elasticsearch_connection_string)
    adapter = ADAPTERS[adapter_name]

    es.indices.create(
        index=index,
        body={
            "settings": {"number_of_shards": 1},
            "mappings": {
                "properties": adapter.get_mapping()
            }
        }
    )

    def get_docs(results):
        id_regex = re.compile("/([a-z0-9]+?)\.jpg$")
        for result in results:
            try:
                result["_id"] = id_regex.search(result["image"]).group(1)
                yield result
            except Exception as e:
                print("Something went wrong!")
                print(e)

    with open(input_file) as json_str:
        input_json = json.load(json_str)
        progress = tqdm(unit="docs", total=len(input_json["results"]))
        for _ in streaming_bulk(
            client=es,
            actions=get_docs(input_json["results"]),
            index=index,
            max_retries=10
        ):
            progress.update(1)


if __name__ == "__main__":
    index_inferrer_output()
