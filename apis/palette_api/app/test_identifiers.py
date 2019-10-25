import requests

from .identifiers import (miro_id_to_catalogue_url,
                          miro_id_to_identifiers, miro_id_to_miro_url)


def test_miro_id_to_miro_url_does_transform():
    target_url = 'https://iiif.wellcomecollection.org/image/A0000001.jpg/full/960,/0/default.jpg'
    result_url = miro_id_to_miro_url('A0000001')
    assert result_url == target_url


def test_miro_id_to_miro_url_returns_valid_url():
    url = miro_id_to_miro_url('A0000001')
    response = requests.get(url)
    assert response.status_code == 200


def test_miro_id_to_catalogue_url_does_transform():
    target_url = 'https://wellcomecollection.org/works/grf79pvz'
    result_url = miro_id_to_catalogue_url('A0000001')
    assert result_url == target_url


def test_miro_id_to_catalogue_url_returns_valid_url():
    url = miro_id_to_catalogue_url('A0000001')
    response = requests.get(url)
    assert response.status_code == 200


def test_miro_id_to_identifiers():
    target = {
        'miro_id': 'A0000001',
        'work_id': 'grf79pvz',
        'miro_url': 'https://iiif.wellcomecollection.org/image/A0000001.jpg/full/960,/0/default.jpg',
        'work_url': 'https://wellcomecollection.org/works/grf79pvz'
    }
    result = miro_id_to_identifiers('A0000001')
    assert result == target
