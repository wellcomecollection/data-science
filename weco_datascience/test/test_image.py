from pathlib import Path

import pytest
from PIL import Image
from weco_datascience.http import (
    close_persistent_client_session,
    start_persistent_client_session,
)
from weco_datascience.image import (
    get_image_from_url,
    get_image_url_from_iiif_url,
)


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

    result = get_image_url_from_iiif_url(
        "https://iiif.wellcomecollection.org/image/V0002882.jpg/info.json"
    )
    expected = (
        "https://iiif.wellcomecollection.org/image/V0002882.jpg/"
        "full/224,224/0/default.jpg"
    )
    assert result == expected


def test_iiif_parse_invalid():
    test_url = "https://example.com/pictures/123/skateboard.png"
    with pytest.raises(ValueError):
        get_image_url_from_iiif_url(test_url)


@pytest.mark.asyncio
async def test_get_local_image():
    file_path = (Path(__file__).parent / "V0050680.jpg").absolute()
    assert file_path.exists()
    image = await get_image_from_url("file://" + str(file_path))
    assert isinstance(image, Image.Image)


@pytest.mark.asyncio
async def test_get_remote_image():
    start_persistent_client_session()
    image = await get_image_from_url(
        "https://iiif.wellcomecollection.org/image/V0050680.jpg/"
        "full/224,224/0/default.jpg"
    )
    await close_persistent_client_session()
    assert isinstance(image, Image.Image)


@pytest.mark.asyncio
async def test_get_iiif_image():
    start_persistent_client_session()

    image = await get_image_from_url(
        "https://iiif.wellcomecollection.org/image/V0050680.jpg/info.json"
    )
    await close_persistent_client_session()
    assert isinstance(image, Image.Image)


@pytest.mark.asyncio
async def test_fails_nonexistant_local_image():
    start_persistent_client_session()
    query_url = str(Path.cwd() / "V0050680.jpg")
    with pytest.raises(ValueError):
        await get_image_from_url(query_url)
    await close_persistent_client_session()


@pytest.mark.asyncio
async def test_fails_nonexistant_remote_image():
    start_persistent_client_session()
    with pytest.raises(ValueError):
        await get_image_from_url("this isn't a url")
    await close_persistent_client_session()
