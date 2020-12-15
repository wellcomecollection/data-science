import pytest
from weco_datascience.api.image import (get_image, get_random_image,
                                        image_search)

from . import image_id


def test_image_search_rejects_invalid_query_param():
    with pytest.raises(ValueError):
        image_search(invalid_param="something")


def test_get_image_rejects_invalid_image_id():
    with pytest.raises(ValueError):
        get_image(query_id="something")


def test_get_image_returns_json():
    json_response = get_image(query_id=image_id)
    assert isinstance(json_response, dict)
    assert json_response["id"] == image_id


def test_get_random_image_returns_json():
    json_response = get_random_image()
    assert isinstance(json_response, dict)
