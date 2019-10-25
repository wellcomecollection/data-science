import numpy as np
import time

import pytest
import requests
from starlette.testclient import TestClient

from .api import app, id_to_url, ids

client = TestClient(app)


def test_health_check():
    response = client.get('/image_similarity/health_check')
    assert response.status_code == 200
    assert response.json() == {'status': 'healthy'}


def test_docs():
    response = client.get('/image_similarity/docs')
    assert response.status_code == 200


def test_redoc():
    response = client.get('/image_similarity/redoc')
    assert response.status_code == 200


def test_id_to_url_does_transform():
    assert id_to_url(
        'A0000001') == 'https://iiif.wellcomecollection.org/image/A0000001.jpg/full/960,/0/default.jpg'


def test_id_to_url_returns_valid_url():
    url = id_to_url('A0000001')
    response = requests.get(url)
    assert response.status_code == 200


def test_response_time():
    n, times = 100, []
    for _ in range(n):
        start_time = time.time()
        client.get('/image_similarity')
        times.append(time.time() - start_time)
    assert np.mean(times) <= 0.1


def test_reponse_contains_all_elements():
    response_elements = [
        'original_image_id',
        'original_image_url',
        'neighbour_ids',
        'neighbour_urls'
    ]
    response = client.get('/image_similarity').json()
    for element in response_elements:
        assert element in response


def test_response_length():
    for i in range(20):
        response = client.get('/image_similarity?n=' + str(i)).json()
        assert len(response['neighbour_ids']) == i
        assert len(response['neighbour_urls']) == i


def test_response_accepts_image_ids():
    for image_id in np.random.choice(ids, size=20):
        response = client.get('/image_similarity?image_id=' + image_id).json()
        assert response['original_image_id'] == image_id


def test_neighbour_list_does_not_contain_original_id():
    for image_id in np.random.choice(ids, size=20):
        response = client.get('/image_similarity?image_id=' + image_id).json()
        assert image_id not in response['neighbour_ids']


def test_does_not_accept_invalid_image_id():
    response = client.get('/image_similarity?image_id=some_rubbish')
    assert response.status_code == 404
    assert response.json() == {'detail': 'Invalid image_id'}


def test_does_not_accept_invalid_n():
    response = client.get('/image_similarity?n=some_rubbish')
    assert response.status_code == 422
