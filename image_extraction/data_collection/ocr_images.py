import csv
import asyncio
from pathlib import Path

import aiohttp
import typer
from weco_datascience.logging import get_logger
from weco_datascience.http import (close_persistent_client_session,
                                   start_persistent_client_session)

from src.images import save_image
from src.wellcome_api import get_manifest_url_from_work_id
from src.iiif import get_ocr_images_from_manifest

log = get_logger(__name__)

async def get_images(work_id):
    data_dir = Path("/data")
    save_dir = data_dir / "ocr_images" / work_id
    ledger_path = save_dir / "ledger.json"

    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        try:
            manifest_url = await get_manifest_url_from_work_id(work_id)
            image_urls = await get_ocr_images_from_manifest(manifest_url)
        except ValueError:
            log.info(f"Couldn't find image urls for {work_id}")
            image_urls = []
        for image_url in image_urls:
            await save_image(image_url, save_dir, ledger_path)


async def main(work_id=None, work_id_path=None):
    if work_id:
        work_ids = [work_id]

    elif work_id_path:
        if work_id_path.exists():
            with open(work_id_path, "r") as f:
                work_ids = list(next(csv.reader(f)))
        else:
            raise ValueError(f"{work_id_path} is not a valid path")
    else:
        raise ValueError(
            "Must supply a work id or a path to a comma separated list of work ids"
        )

    start_persistent_client_session()
    for work_id in work_ids:
        await get_images(work_id)
    await close_persistent_client_session()


def run_async(
    work_id: str = typer.Option(
        None,
        help="the ID of the work whose images we'll fetch"
    ),
    work_id_path: Path = typer.Option(
        None,
        help="Path to a list of comma-separated work ids"
    ),
):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(work_id, work_id_path))


if __name__ == "__main__":
    typer.run(run_async)
