import hashlib
import json
from pathlib import Path

import httpx
from weco_datascience.image import get_image_from_url
from weco_datascience.logging import get_logger

log = get_logger(__name__)


def generate_file_name(image_url):
    file_name = hashlib.sha256(image_url.encode()).hexdigest()
    file_extension = Path(image_url).suffix
    return file_name + file_extension


def update_ledger(image_url, ledger_path):
    if Path(ledger_path).exists():
        with open(ledger_path, "r") as f:
            ledger = json.load(f)
    else:
        log.info(f"creating new ledger at {ledger_path}")
        ledger_path.touch()
        ledger = {}

    ledger[image_url] = generate_file_name(image_url)
    with open(ledger_path, "w+") as f:
        json.dump(ledger, f)


async def save_image(image_url, save_dir, ledger_path):
    log.info(f"Downloading image from {image_url}")
    image = await get_image_from_url(image_url)
    try:
        file_name = generate_file_name(image_url)
        save_path = Path(save_dir) / file_name
        image.save(save_path)
        log.info(f"Saving image at {save_path}")
        try:
            update_ledger(image_url, ledger_path)
        except:
            raise ValueError("couldn't update the ledger")
    except:
        raise ValueError("couldn't save the image")
