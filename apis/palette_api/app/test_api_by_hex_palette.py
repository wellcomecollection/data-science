import json
import time

import numpy as np
import pytest
from starlette.testclient import TestClient

from .api import app, image_ids
from .utils import random_hex

client = TestClient(app)


def test_response_time():
    n, times = 100, []
    for _ in range(n):
        start_time = time.time()
        client.get('/palette_api/by_palette')
        times.append(time.time() - start_time)
    assert np.mean(times) <= 0.1


def test_reponse_contains_all_elements():
    response_elements = [
        'palette',
        'neighbour_ids',
        'neighbour_urls'
    ]
    response = client.get('/palette_api/by_palette').json()
    for element in response_elements:
        assert element in response


def test_response_length():
    for i in range(20):
        response = client.get(
            '/palette_api/by_palette?n=' + str(i)).json()
        assert len(response['neighbour_ids']) == i
        assert len(response['neighbour_urls']) == i


def test_does_not_accept_invalid_n():
    response = client.get(
        '/palette_api/by_palette?n=some_rubbish')
    assert response.status_code == 422


def test_response_accepts_valid_palette():
    for _ in range(20):
        palette = [random_hex() for _ in range(5)]
        str_palette = json.dumps(palette)
        response = client.get(
            '/palette_api/by_palette?palette=' + str_palette).json()
        assert response['palette'] == palette


def test_response_rejects_short_palette():
    palette = [random_hex() for _ in range(4)]
    str_palette = json.dumps(palette)
    response = client.get(
        '/palette_api/by_palette?palette=' + str_palette)
    assert response.status_code == 500
    assert response.json() == {
        "detail": "Palette must consist of 5 colours"
    }


def test_response_rejects_long_palette():
    palette = [random_hex() for _ in range(6)]
    str_palette = json.dumps(palette)
    response = client.get(
        '/palette_api/by_palette?palette=' + str_palette)
    assert response.status_code == 500
    assert response.json() == {
        "detail": "Palette must consist of 5 colours"
    }


def test_response_rejects_invalid_hex_colour():
    palette = [random_hex() for _ in range(4)] + ['an_invalid_hex_colour']
    str_palette = json.dumps(palette)
    response = client.get(
        '/palette_api/by_palette?palette=' + str_palette)
    assert response.status_code == 500
    assert response.json() == {
        "detail": "an_invalid_hex_colour is not a valid hex colour"
    }
