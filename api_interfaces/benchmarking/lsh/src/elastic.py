import os
from elasticsearch import Elasticsearch, helpers


class ES:
    def __init__(self):
        self.es = Elasticsearch(
            hosts=[{
                'host': os.environ["ES_HOST"],
                'port': os.environ["ES_PORT"]
            }],
            http_auth=(
                os.environ["ES_USER"],
                os.environ["ES_PASS"]
            )
        )

    def create_index(self, index_name, properties):
        try:
            self.es.indices.delete(index_name)
        except:
            pass

        print(f"Creating index: {index_name}")
        self.es.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": properties
                }
            }
        )

    def index_document(self, index_name, body):
        self.es.index(
            index=index_name,
            body=body
        )

    def bulk_index_documents(self, gendata):
        """
        gendata should be a generator of dicts matching the index mapping
        """
        print("Indexing documents")
        helpers.bulk(
            self.es,
            gendata,
            request_timeout=600,
            chunk_size=50
        )

    def lsh_query(self, index_name, image_id, n):
        query = {
            "query": {
                "more_like_this": {
                    "fields": ["lsh_features"],
                    "like": {
                        "_index": index_name,
                        "_id": image_id
                    },
                    "min_term_freq": 1,
                    "max_query_terms": 1000,
                    "minimum_should_match": 1
                }
            },
            "stored_fields": [],
            "size": n
        }
        search_response = self.es.search(
            index=index_name,
            body=query
        )
        try:
            result_ids = [
                hit['_id'] for hit in search_response['hits']['hits']
            ][:n]
        except IndexError:
            result_ids = []

        return result_ids

    def exact_query(self, image_id, n):
        get_response = self.es.get(index="feature_vectors", id=image_id)
        search_response = self.es.search(
            index="feature_vectors",
            body={
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": (
                                "cosineSimilarity(params.query_features_1, 'features_1') + "
                                "cosineSimilarity(params.query_features_2, 'features_2')"
                            ),
                            "params": {
                                "query_features_1": get_response['_source']['features_1'],
                                "query_features_2": get_response['_source']['features_2']
                            }
                        }
                    }
                },
                "stored_fields": []
            }
        )
        try:
            result_ids = [
                hit['_id'] for hit in search_response['hits']['hits']
            ][:n]
        except IndexError:
            result_ids = []

        return result_ids
