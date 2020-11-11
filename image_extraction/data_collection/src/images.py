import hashlib
import json
from pathlib import Path

from aiofile import AIOFile
from weco_datascience.http import fetch_url_bytes
from weco_datascience.image import get_image_from_url
from weco_datascience.logging import get_logger

log = get_logger(__name__)


def generate_file_name(image_url):
    file_name = hashlib.sha256(image_url.encode()).hexdigest()
    file_extension = Path(image_url).suffix
    return file_name + file_extension


async def update_ledger(image_url, ledger_path):
    if Path(ledger_path).exists():
        async with AIOFile(ledger_path, 'r') as afp:
            ledger_json_string = await afp.read()
        ledger = json.loads(ledger_json_string)
    else:
        log.info(f"Creating new ledger at {ledger_path}")
        ledger_path.touch()
        ledger = {}

    ledger[image_url] = generate_file_name(image_url)
    ledger_json_string = json.dumps(ledger)
    async with AIOFile(ledger_path, 'w') as afp:
        await afp.write(ledger_json_string)


async def save_image(image_url, save_dir, ledger_path=None):
    log.info(f"Downloading image from {image_url}")
    image_response = await fetch_url_bytes(image_url)

    try:
        file_name = generate_file_name(image_url)
        save_path = Path(save_dir) / file_name

        async with AIOFile(save_path, 'wb') as afp:
            log.info(f"Saving image at {save_path}")
            await afp.write(image_response["bytes"])

        if ledger_path:
            try:
                await update_ledger(image_url, ledger_path)
            except:
                raise ValueError("Couldn't update the ledger")
    except:
        raise ValueError("Couldn't save the image")
