from collections import MutableMapping

import pandas as pd
from elasticsearch import Elasticsearch, helpers

from .aws import get_secret_string, get_session


def flatten(d, parent_key="", sep="."):
    """
    flatten a nested dictionary so that it can be more neatly loaded into a
    pandas dataframe

    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys

    Parameters
    ----------
    d: dict
        the dictionary to flatten
    parent_key: str, optional
        in nested dicts, the name of the parent fields
    sep: str, optional
        the string used to concatenate parent and child field names

    Returns
    -------
    d: dict
        the flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def query_es(config, query):
    """
    Run a query against a specified elasticsearch index

    Parameters
    ----------
    config: dict
        elasticsearch username, password, index, and endpoint for the request
    query: dict
        the query, following the elasticsearch json query structure

    Returns
    -------
    df: pd.DataFrame
        a pandas dataframe containing the flattened response data
    """
    client = Elasticsearch(
        config["host"], http_auth=(config["username"], config["password"])
    )
    response = client.search(body=query, index=config["index"])
    data = [flatten(event["_source"]) for event in response["hits"]["hits"]]
    return pd.DataFrame(data)


def get_data_in_date_range(
    config, start_date="now-1d", end_date="now", timestamp_field="@timestamp"
):
    """
    Fetch data within a specified date/time range

    Parameters
    ----------
    config: dict
        elasticsearch username, password, index, and endpoint for the request
    start_date: str, datetime, optional
        defaults to now
    end_date: str, datetime, optional
        defaults to now
    timestamp_field: str, optional
        the timestamp field which should be used to sort the data by recency

    Returns
    -------
    df: pd.DataFrame
        a pandas dataframe containing the flattened response data
    """
    query = {
        "query": {
            "range": {timestamp_field: {"gte": start_date, "lt": end_date}}
        },
        "size": 1_000_000,
    }
    return query_es(config, query)


def get_recent_data(config, n, index, timestamp_field="@timestamp"):
    """
    Fetch the `n` most recent documents from a specified elasticsearch index

    Parameters
    ----------
    config: dict
        elasticsearch username, password, index, and endpoint for the request
    n: int
        the number of documents to return
    timestamp_field: str, optional
        the timestamp field which should be used to sort the data by recency

    Returns
    -------
    df: pd.DataFrame
        a pandas dataframe containing the flattened response data
    """
    client = Elasticsearch(
        config["host"], http_auth=(config["username"], config["password"])
    )
    response = helpers.scan(
        client,
        query={"sort": [{timestamp_field: "desc"}]},
        index=config["index"],
        preserve_order=True,
    )
    data = [flatten(next(response)["_source"]) for _ in range(n)]
    return pd.DataFrame(data)


def get_es_client():
    """
    Returns an Elasticsearch client with read-only access to the
    reporting cluster.
    """
    session = get_session(
        role_arn="arn:aws:iam::760097843905:role/platform-developer"
    )

    host = get_secret_string(session, secret_id="reporting/es_host")
    username = get_secret_string(session, secret_id="reporting/read_only/es_username")
    password = get_secret_string(session, secret_id="reporting/read_only/es_password")

    return Elasticsearch(f"https://{username}:{password}@{host}:9243")
