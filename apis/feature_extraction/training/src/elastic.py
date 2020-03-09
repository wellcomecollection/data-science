import os
import numpy as np
from elasticsearch import Elasticsearch, helpers


def get_all_ids(es_client, index_name):
    response = helpers.scan(
        client=es_client,
        index=index_name,
        query={
            "query": {
                "match_all": {}
            },
            "stored_fields": []
        }
    )

    return [hit['_id'] for hit in list(response)]


def get_random_documents(es_client, index_name, n):
    all_ids = get_all_ids(es_client, index_name)
    if len(all_ids) > n:
        query_ids = np.random.choice(all_ids, n, replace=False).tolist()
    else:
        query_ids = all_ids

    response = es_client.mget(
        index=index_name,
        body={'ids': query_ids}
    )

    return response


def get_random_feature_vectors(es_client, index_name, n):
    response = get_random_documents(es_client, index_name, n)
    docs = [doc['_source']['doc'] for doc in response['docs']]
    feature_vectors = np.stack([
        np.concatenate([doc['features_1'], doc['features_2']], axis=0)
        for doc in docs
    ])
    return feature_vectors
