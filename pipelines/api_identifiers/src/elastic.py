import os
from elasticsearch import Elasticsearch


def get_es_client():
    es_url = os.environ['ES_URL']
    es_username = os.environ['ES_USERNAME']
    es_password = os.environ['ES_PASSWORD']
    return Elasticsearch(
        es_url, http_auth=(es_username, es_password)
    )
