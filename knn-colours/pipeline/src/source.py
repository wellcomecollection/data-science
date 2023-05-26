from elasticsearch.helpers import scan
from .elasticsearch import get_catalogue_elastic_client

def yield_source_images(pipeline_date):
    es = get_catalogue_elastic_client(pipeline_date)
    index_name = f"images-indexed-{pipeline_date}"

    for image_data in scan(es, index=index_name):
        yield image_data["_source"]["display"]


def count_source_images(pipeline_date):
    es = get_catalogue_elastic_client(pipeline_date)
    index_name = f"images-indexed-{pipeline_date}"
    return es.count(index=index_name)["count"]
