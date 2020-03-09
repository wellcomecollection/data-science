import os
import numpy as np
from elasticsearch import Elasticsearch, helpers


def get_random_documents(n_documents, es_client):
    scan_response = helpers.scan(
        client=es_client,
        index='test-index',
        query={
            "query": {
                "match_all": {}
            },
            "stored_fields": []
        }
    )

    all_ids = [hit['_id'] for hit in list(scan_response)]

    if len(all_ids) > n_documents:
        query_ids = np.random.choice(
            all_ids, n_documents, replace=False).tolist()
    else:
        query_ids = all_ids

    documents = es_client.mget(
        index='test-index',
        body={'ids': query_ids}
    )

    return documents


def get_random_feature_vectors(n_documents, es_client=None):
    es_client = es_client or Elasticsearch(
        host=os.environ['ES_HOST'],
        http_auth=(os.environ['ES_USERNAME'], os.environ['ES_PASSWORD'])
    )

    documents = get_random_documents(n_documents, es_client)
    docs = [doc['_source']['doc'] for doc in documents['docs']]
    feature_vectors = np.stack([
        np.concatenate(
            [doc['feature_vector_1'], doc['feature_vector_2']],
            axis=0
        ) for doc in docs
    ])
    return feature_vectors
