import numpy as np
import requests
from .utils import id_to_url, get_path_indexes, get_ideal_coords


def test_id_to_url_does_transform():
    assert id_to_url(
        'A0000001') == 'https://iiif.wellcomecollection.org/image/A0000001.jpg/full/960,/0/default.jpg'


def test_id_to_url_returns_valid_url():
    url = id_to_url('A0000001')
    response = requests.get(url)
    assert response.status_code == 200


def test_get_path_indexes():
    closest_indexes = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ])
    start_index = 0
    end_index = 10

    pathway = get_path_indexes(closest_indexes, start_index, end_index)

    assert pathway[0] == start_index
    assert pathway[-1] == end_index
    assert pathway == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_get_ideal_coords():
    n = 10
    start_coord = np.random.random(size=1)
    end_coord = np.random.random(size=1)
    ideal_coords = np.vstack([
        [start_coord],
        get_ideal_coords(start_coord, end_coord, n),
        [end_coord]
    ])
    index = np.random.randint(1, n - 1)
    expected_value = ((index) * (end_coord - start_coord) / n) + start_coord
    assert np.isclose(ideal_coords[index], expected_value, atol=0.05).all()
