import urllib
from gzip import GzipFile
from io import BytesIO
from pathlib import Path

import requests
from urlpath import URL

snapshot_url = "https://data.wellcomecollection.org/catalogue/v2/works.json.gz"


def get_snapshot(save_dir, unzip=True):
    save_dir = Path(save_dir).expanduser().resolve()
    file_path = save_dir / "works.json.gz"
    urllib.request.urlretrieve(snapshot_url, file_path)
    if unzip:
        with GzipFile(file_path, 'rb') as f:
            data = f.read()

        decompressed_file_path = save_dir / "works.json"
        decompressed_file_path.touch()
        with open(decompressed_file_path, 'wb') as f:
            f.write(data)

        # delete the original .gz file
        file_path.unlink()
