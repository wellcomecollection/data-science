import pytest
from weco_datascience.api.catalogue import (get_random_work, get_work,
                                            works_search)

from . import work_id


def test_works_search_rejects_invalid_query_param():
    with pytest.raises(ValueError):
        works_search(invalid_param="something")


def test_get_work_rejects_invalid_work_id():
    with pytest.raises(ValueError):
        get_work(query_id="something")


def test_get_work_returns_json():
    json_response = get_work(query_id=work_id)
    assert isinstance(json_response, dict)
    assert json_response["id"] == work_id


def test_get_random_work_returns_json():
    json_response = get_random_work()
    assert isinstance(json_response, dict)
