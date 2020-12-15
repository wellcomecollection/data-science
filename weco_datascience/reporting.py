from elasticsearch import Elasticsearch, helpers, RequestError
import pandas as pd
from collections import MutableMapping


def flatten(nested_dict, parent_key=''):
    items = []
    for k, v in nested_dict.items():
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, k).items())
        else:
            items.append((k, v))
    return dict(items)


def query_es(config, query, index):
    client = Elasticsearch(
        config["host"], http_auth=(config['username'], config['password'])
    )
    response = client.search(body=query, index=index)
    data = [
        flatten(event['_source'])
        for event in response["hits"]["hits"]
    ]
    return pd.DataFrame(data)


def get_recent_data(config, n, index):
    client = Elasticsearch(
        config["host"], http_auth=(config['username'], config['password'])
    )
    try:
        response = helpers.scan(
            client,
            query={"sort": [{"timestamp": "desc"}]},
            index=index,
            preserve_order=True
        )
    except RequestError:
        try:
            response = helpers.scan(
                client,
                query={"sort": [{"@timestamp": "desc"}]},
                index=index,
                preserve_order=True
            )
        except Error as e:
            raise e

    data = [flatten(next(response)["_source"]) for _ in range(n)]
    return pd.DataFrame(data)
