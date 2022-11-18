import os
from time import sleep

from elasticsearch import Elasticsearch

from .log import get_logger

log = get_logger()


def get_local_elastic_client() -> Elasticsearch:
    es = Elasticsearch(
        "http://elasticsearch:9200",
        basic_auth=("elastic", "password"),
        request_timeout=3600,
    )
    wait_for_client(es)
    return es


def get_concepts_prototype_elastic_client() -> Elasticsearch:
    es = Elasticsearch(
        hosts=os.environ.get("ES_PROTOTYPE_HOST"),
        basic_auth=(
            os.environ.get("ES_PROTOTYPE_USERNAME"),
            os.environ.get("ES_PROTOTYPE_PASSWORD"),
        ),
        request_timeout=3600,
    )
    wait_for_client(es)
    return es


def wait_for_client(es: Elasticsearch):
    while True:
        try:
            es.ping()
            break
        except Exception:
            log.info("Waiting for elasticsearch to start...")
            sleep(3)
