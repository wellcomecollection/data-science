import re

import numpy as np
import requests
from hypothesis import given
from hypothesis import strategies as st

from .utils import embed, id_to_url


def test_id_to_url_does_transform():
    assert id_to_url(
        'A0000001') == 'https://iiif.wellcomecollection.org/image/A0000001.jpg/full/960,/0/default.jpg'


def test_id_to_url_returns_valid_url():
    url = id_to_url('A0000001')
    response = requests.get(url)
    assert response.status_code == 200


@given(st.text())
def test_embed_returns_valid_array(text):
    embedded = embed(text)
    assert isinstance(embedded, np.ndarray)
    assert embedded.shape == (4096,)
