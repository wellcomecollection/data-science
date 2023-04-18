import os
from time import sleep

from elasticsearch import Elasticsearch

from .log import get_logger
from .secrets import get_secret
import httpx

log = get_logger()


def wait_for_client(es: Elasticsearch):
    while True:
        try:
            es.ping()
            break
        except Exception:
            log.info("Waiting for elasticsearch to start...")
            sleep(3)


def get_elastic_client() -> Elasticsearch:
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


def get_reporting_es_client():
    host = get_secret("reporting/es_host")
    es_password = get_secret("reporting/read_only/es_password")
    es_username = get_secret("reporting/read_only/es_username")
    reporting_es_client = Elasticsearch(
        hosts=f"https://{host}:443",
        basic_auth=(es_username, es_password),
        timeout=30,
        retry_on_timeout=True,
        max_retries=10,
    )
    wait_for_client(reporting_es_client)
    return reporting_es_client


def get_pipeline_es_client():
    index_name = httpx.get(
        "https://api.wellcomecollection.org/catalogue/v2/_elasticConfig"
    ).json()["worksIndex"]
    index_date = index_name.replace("works-indexed-", "")
    secret_prefix = f"elasticsearch/pipeline_storage_{index_date}/"
    public_host = get_secret(secret_prefix + "public_host")
    port = get_secret(secret_prefix + "port")
    protocol = get_secret(secret_prefix + "protocol")
    host = f"{protocol}://{public_host}:{port}"

    es_password = get_secret(secret_prefix + "es_password")
    es_username = get_secret(secret_prefix + "es_username")
    pipeline_es_client = Elasticsearch(
        hosts=host,
        basic_auth=(es_username, es_password),
        timeout=30,
        retry_on_timeout=True,
        max_retries=10,
    )
    wait_for_client(pipeline_es_client)
    return pipeline_es_client
