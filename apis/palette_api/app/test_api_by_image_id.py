import time

import numpy as np
import pytest
from starlette.testclient import TestClient

from .api import app, image_ids

client = TestClient(app)


def test_response_time():
    n, times = 100, []
    for _ in range(n):
        start_time = time.time()
        client.get('/palette_api/by_image_id')
        times.append(time.time() - start_time)
    assert np.mean(times) <= 0.1


def test_reponse_contains_all_elements():
    response_elements = [
        'original_image_id',
        'original_image_url',
        'neighbour_ids',
        'neighbour_urls'
    ]
    response = client.get('/palette_api/by_image_id').json()
    for element in response_elements:
        assert element in response


def test_response_length():
    for i in range(20):
        response = client.get(
            '/palette_api/by_image_id?n=' + str(i)).json()
        assert len(response['neighbour_ids']) == i
        assert len(response['neighbour_urls']) == i


def test_response_accepts_image_ids():
    for image_id in np.random.choice(image_ids, size=20):
        response = client.get(
            '/palette_api/by_image_id?image_id=' + image_id).json()
        assert response['original_image_id'] == image_id


def test_neighbour_list_does_not_contain_original_id():
    for image_id in np.random.choice(image_ids, size=20):
        response = client.get(
            '/palette_api/by_image_id?image_id=' + image_id).json()
        assert image_id not in response['neighbour_ids']


def test_does_not_accept_invalid_image_id():
    response = client.get(
        '/palette_api/by_image_id?image_id=some_rubbish')
    assert response.status_code == 404
    assert response.json() == {'detail': 'Invalid image_id'}


def test_does_not_accept_invalid_n():
    response = client.get(
        '/palette_similarity/by_image_id?n=some_rubbish')
    assert response.status_code == 422
