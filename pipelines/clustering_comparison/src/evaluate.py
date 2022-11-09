from numpy import ndarray as Array
from scipy.spatial.distance import cdist

from elasticsearch import Elasticsearch


def calculate_exact_distances(query_vector: Array, features: Array) -> Array:
    """Calculate the exact distance between a query vector and all feature
    vectors.

    Args:
        query_vector (np.ndarray): The query vector.
        features (np.ndarray): The feature vectors.

    Returns:
        np.ndarray: The cosine distances between the query vector and all
        feature vectors.
    """
    return cdist(query_vector, features, metric="cosine")


def calculate_lsh_distances(
    query_string: str, client: Elasticsearch, index: str, n: int = 10
) -> Array:
    """
    Calculate the approximate distance between a query vector and all feature
    vectors using a more_like_this query of locality-sensitive hashes in
    elasticsearch

    Args:
        query_string (str): The query string.
        client (Elasticsearch): The elasticsearch client.
        index (str): The index to search.
        n (int, optional): The number of nearest neighbors to return. Defaults
        to 10.

    Returns:
        list: The approximate nearest neighbors.
    """
    res = client.search(
        index=index,
        body={
            "query": {
                "more_like_this": {
                    "like": query_string,
                    "fields": ["lsh"],
                    "min_term_freq": 1,
                    "max_query_terms": 1000,
                    "min_doc_freq": 1,
                }
            }
        },
    )
    return [res["hits"]["hits"][i]["_score"] for i in range(n)]
