from pathlib import Path

import pytest
from PIL.Image import Image
from weco_datascience.http import (close_persistent_client_session,
                                   start_persistent_client_session)
from weco_datascience.image import (get_image_from_url,
                                    get_image_url_from_iiif_url,
                                    is_valid_image)
from weco_datascience.http import fetch_url_bytes
from . import iiif_url, image_url, local_image_path, invalid_url


def test_iiif_parse_dlcs():
    result = get_image_url_from_iiif_url(
        "https://dlcs.io/iiif-img/wellcome/5/b28047345_0009.jp2/info.json"
    )
    expected = (
        "https://dlcs.io/thumbs/wellcome/5/b28047345_0009.jp2/"
        "full/!400,400/0/default.jpg"
    )
    assert result == expected


def test_iiif_parse_other():
    result = get_image_url_from_iiif_url(iiif_url)
    assert result == image_url


def test_iiif_parse_invalid():
    with pytest.raises(ValueError):
        get_image_url_from_iiif_url(invalid_url)


@pytest.mark.asyncio
async def test_is_valid_image():
    start_persistent_client_session()
    response = await fetch_url_bytes(image_url)
    await close_persistent_client_session()
    assert is_valid_image(response["object"])


@pytest.mark.asyncio
async def test_is_invalid_image():
    start_persistent_client_session()
    response = await fetch_url_bytes(invalid_url)
    await close_persistent_client_session()
    assert not is_valid_image(response["object"])


@pytest.mark.asyncio
async def test_get_local_image():
    image = await get_image_from_url(local_image_path)
    assert isinstance(image, Image)


@pytest.mark.asyncio
async def test_get_remote_image():
    start_persistent_client_session()
    image = await get_image_from_url(image_url)
    await close_persistent_client_session()
    assert isinstance(image, Image)


@pytest.mark.asyncio
async def test_get_iiif_image():
    start_persistent_client_session()
    image = await get_image_from_url(iiif_url)
    await close_persistent_client_session()
    assert isinstance(image, Image)


@pytest.mark.asyncio
async def test_fails_nonexistant_local_image():
    start_persistent_client_session()
    query_url = "nonexistent_image.jpg"
    with pytest.raises(ValueError):
        await get_image_from_url(query_url)
    await close_persistent_client_session()


@pytest.mark.asyncio
async def test_fails_nonexistant_remote_image():
    start_persistent_client_session()
    with pytest.raises(ValueError):
        await get_image_from_url("this isn't a url")
    await close_persistent_client_session()
