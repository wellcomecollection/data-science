import os
from time import sleep

from elasticsearch import Elasticsearch

from .log import get_logger

log = get_logger()


def get_elastic_client() -> Elasticsearch:
    es = Elasticsearch(
        hosts=os.environ.get("ES_HOST"),
        basic_auth=(
            os.environ.get("ES_USERNAME"),
            os.environ.get("ES_PASSWORD"),
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
