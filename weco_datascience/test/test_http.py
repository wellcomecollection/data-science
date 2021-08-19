import json

import pytest

from weco_datascience.http import (
    close_persistent_client_session,
    fetch_redirect_url,
    fetch_url_bytes,
    fetch_url_json,
    start_persistent_client_session,
)

from . import iiif_url, image_url


@pytest.mark.asyncio
async def test_get_bytes():
    start_persistent_client_session()
    response = await fetch_url_bytes(image_url)
    await close_persistent_client_session()
    assert response["object"].status == 200
    assert isinstance(response["bytes"], bytes)


@pytest.mark.asyncio
async def test_get_json():
    start_persistent_client_session()
    response = await fetch_url_json(iiif_url)
    await close_persistent_client_session()
    assert response["object"].status == 200
    assert json.dumps(response["json"])


@pytest.mark.asyncio
async def test_bad_json():
    start_persistent_client_session()
    with pytest.raises(ValueError):
        await fetch_url_json(image_url)
    await close_persistent_client_session()


@pytest.mark.asyncio
async def test_redirect():
    start_persistent_client_session()
    original_url = "https://id.loc.gov/authorities/label/Physical%20geography"
    response = await fetch_redirect_url(original_url)
    await close_persistent_client_session()
    assert response["url"] != original_url


@pytest.mark.asyncio
async def test_non_redirect():
    start_persistent_client_session()
    response = await fetch_redirect_url(iiif_url)
    await close_persistent_client_session()
    assert response["url"] == iiif_url
