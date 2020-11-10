import asyncio
from pathlib import Path

import aiohttp
import typer
from weco_datascience.http import (close_persistent_client_session,
                                   start_persistent_client_session)

from src.images import save_image
from src.wellcome_api import get_manifest_url_from_work_id
from src.iiif import get_ocr_images_from_manifest


async def main(work_id):
    data_dir = Path("/data")
    save_dir = data_dir / "ocr_images" / work_id
    ledger_path = data_dir / "ledger.json"

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    start_persistent_client_session()
    manifest_url = await get_manifest_url_from_work_id(work_id)
    image_urls = await get_ocr_images_from_manifest(manifest_url)
    for image_url in image_urls:
        await save_image(image_url, save_dir, ledger_path)
    await close_persistent_client_session()


def run_async(work_id: str = typer.Argument(
        ...,
        help="the ID of the work whose images we'll fetch"
    )
):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(work_id))


if __name__ == "__main__":
    typer.run(run_async)
