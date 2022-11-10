import os
from time import sleep

from elasticsearch import Elasticsearch

from .log import get_logger
from .secrets import get_secret
from .wellcome import pipeline_date

log = get_logger()


def get_catalogue_elastic_client() -> Elasticsearch:
    secret_prefix = f"elasticsearch/pipeline_storage_{pipeline_date}/"
    es_password = get_secret(secret_prefix + "es_password")
    es_username = get_secret(secret_prefix + "es_username")
    protocol = get_secret(secret_prefix + "protocol")
    public_host = get_secret(secret_prefix + "public_host")
    port = get_secret(secret_prefix + "port")

    es = Elasticsearch(
        f"{protocol}://{public_host}:{port}",
        basic_auth=(es_username, es_password),
    )
    wait_for_client(es)
    return es


def get_concepts_elastic_client() -> Elasticsearch:
    es_password = os.environ.get("concepts_password")
    es_username = os.environ.get("concepts_username")
    host = os.environ.get("concepts_host")

    es = Elasticsearch(hosts=host, basic_auth=(es_username, es_password))
    wait_for_client(es)
    return es


def get_rank_elastic_client() -> Elasticsearch:
    secret_prefix = "elasticsearch/rank/"
    es_password = get_secret(secret_prefix + "ES_RANK_PASSWORD")
    es_username = get_secret(secret_prefix + "ES_RANK_USER")
    cloud_id = get_secret(secret_prefix + "ES_RANK_CLOUD_ID")
    es = Elasticsearch(
        cloud_id=cloud_id,
        basic_auth=(es_username, es_password),
    )
    wait_for_client(es)
    return es


def get_local_elastic_client() -> Elasticsearch:
    es = Elasticsearch(
        "http://elasticsearch:9200",
        basic_auth=("elastic", "password"),
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
