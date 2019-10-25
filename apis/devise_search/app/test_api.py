import time

import numpy as np
import pytest
import requests
from hypothesis import given
from hypothesis import strategies as st
from starlette.testclient import TestClient

from .api import app, search_index

client = TestClient(app)


def test_health_check():
    response = client.get('/devise_search/health_check')
    assert response.status_code == 200
    assert response.json() == {'status': 'healthy'}


def test_docs():
    response = client.get('/devise_search/docs')
    assert response.status_code == 200


def test_redoc():
    response = client.get('/devise_search/redoc')
    assert response.status_code == 200


def test_response_time():
    n, times = 100, []
    for _ in range(n):
        start_time = time.time()
        client.get('/devise_search')
        times.append(time.time() - start_time)
    assert np.mean(times) <= 0.1


def test_reponse_contains_all_elements():
    response_elements = [
        'query_text',
        'neighbour_ids',
        'neighbour_urls'
    ]
    response = client.get('/devise_search').json()
    for element in response_elements:
        assert element in response


def test_response_length():
    for i in range(20):
        response = client.get('/devise_search?n=' + str(i)).json()
        assert len(response['neighbour_ids']) == i
        assert len(response['neighbour_urls']) == i


def test_does_not_accept_invalid_n():
    response = client.get('/devise_search?n=some_rubbish')
    assert response.status_code == 422


def test_hnsw_index_accepts_array():
    query_embedding = np.random.random(size=4096)
    neighbour_indexes, _ = search_index.knnQuery(query_embedding, 10)
    assert isinstance(neighbour_indexes, np.ndarray)
    assert isinstance(neighbour_indexes[0], np.int32)
    assert len(neighbour_indexes) == 10
