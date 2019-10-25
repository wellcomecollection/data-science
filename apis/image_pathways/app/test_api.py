import numpy as np
import time

import pytest
import requests
from starlette.testclient import TestClient

from .api import app
from .utils import ids

client = TestClient(app)


def test_health_check():
    response = client.get('/image_pathways/health_check')
    assert response.status_code == 200
    assert response.json() == {'status': 'healthy'}


def test_docs():
    response = client.get('/image_pathways/docs')
    assert response.status_code == 200


def test_redoc():
    response = client.get('/image_pathways/redoc')
    assert response.status_code == 200


def test_response_time():
    n, times = 100, []
    for _ in range(n):
        start_time = time.time()
        client.get('/image_pathways')
        times.append(time.time() - start_time)
    assert np.mean(times) <= 0.5


def test_reponse_contains_all_elements():
    response_elements = [
        'id_path',
        'image_url_path'
    ]
    response = client.get('/image_pathways').json()
    for element in response_elements:
        assert element in response


def test_response_length():
    for path_length in range(3, 25):
        response = client.get(
            '/image_pathways?path_length=' + str(path_length)
        ).json()
        assert len(response['id_path']) == path_length
        assert len(response['image_url_path']) == path_length


def test_response_accepts_image_id_1():
    for image_id in np.random.choice(ids, size=10):
        response = client.get('/image_pathways?id_1=' + image_id)
        assert response.status_code == 200


def test_response_accepts_image_id_2():
    for image_id in np.random.choice(ids, size=10):
        response = client.get('/image_pathways?id_2=' + image_id)
        assert response.status_code == 200


def test_does_not_accept_invalid_image_id_1():
    response = client.get('/image_pathways?id_1=some_rubbish')
    assert response.status_code == 404
    assert response.json() == {'detail': 'Invalid image_id'}


def test_does_not_accept_invalid_image_id_2():
    response = client.get('/image_pathways?id_2=some_rubbish')
    assert response.status_code == 404
    assert response.json() == {'detail': 'Invalid image_id'}


def test_does_not_accept_invalid_n():
    response = client.get('/image_pathways?path_length=some_rubbish')
    assert response.status_code == 422
