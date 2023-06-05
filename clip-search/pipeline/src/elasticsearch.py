import os
from time import sleep

from elasticsearch import Elasticsearch

from .log import get_logger
from .secrets import get_secret

log = get_logger()


def get_catalogue_elastic_client(pipeline_date) -> Elasticsearch:
    secret_prefix = f"elasticsearch/pipeline_storage_{pipeline_date}/"
    es_password = get_secret(secret_prefix + "es_password")
    es_username = get_secret(secret_prefix + "es_username")
    protocol = get_secret(secret_prefix + "protocol")
    public_host = get_secret(secret_prefix + "public_host")
    port = get_secret(secret_prefix + "port")

    es = Elasticsearch(
        f"{protocol}://{public_host}:{port}",
        basic_auth=(es_username, es_password),
        timeout=30,
    )
    wait_for_client(es)
    return es


def get_prototype_elastic_client() -> Elasticsearch:
    environment = os.environ.get("ENVIRONMENT", "local")

    if environment == "aws":
        cloud_id = os.environ["CLOUD_ID"]
        es_username = os.environ["USERNAME"]
        es_password = os.environ["PASSWORD"]

    elif environment == "local":
        secret_prefix = "elasticsearch/concepts-prototype/"
        cloud_id = get_secret(secret_prefix + "CLOUD_ID")
        es_password = get_secret(secret_prefix + "PASSWORD")
        es_username = get_secret(secret_prefix + "USERNAME")

    else:
        raise ValueError(f"Unknown environment: {environment}")

    es = Elasticsearch(
        cloud_id=cloud_id,
        basic_auth=(es_username, es_password),
        timeout=30,
        retry_on_timeout=True,
        max_retries=10,
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
