import gzip
import json
import shutil
from pathlib import Path

from piffle.image import IIIFImageClient
import httpx
from tqdm import tqdm


def load_records(path):
    """iterate over a large dataset with yield"""
    with open(path, 'r', encoding='utf-8') as f:
        while line := f.readline():
            yield json.loads(line)


data_dir = Path(__file__).parent / "data"
if not data_dir.exists():
    data_dir.mkdir()

dataset_url = "https://data.wellcomecollection.org/catalogue/v2/works.json.gz"
filename = Path(dataset_url).name
zipped_works_file_path = data_dir / filename
works_file_path = data_dir / zipped_works_file_path.stem

if not works_file_path.exists():
    print(f"Downloading works dataset to {works_file_path.absolute()}...")
    if not zipped_works_file_path.exists():
        with open(zipped_works_file_path, "wb") as download_file:
            with httpx.stream("GET", dataset_url, timeout=999999) as response:
                total = int(response.headers["Content-Length"])
                with tqdm(
                    total=total,
                    unit_scale=True,
                    unit_divisor=1024,
                    unit="B",
                    desc=filename,
                ) as progress:
                    num_bytes_downloaded = response.num_bytes_downloaded
                    for chunk in response.iter_bytes():
                        download_file.write(chunk)
                        progress.update(
                            response.num_bytes_downloaded - num_bytes_downloaded
                        )
                        num_bytes_downloaded = response.num_bytes_downloaded
    
    print("Unzipping works dataset...")
    with gzip.open(zipped_works_file_path, "rb") as f_in:
        with open(works_file_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    print("Deleting zipped works dataset...")
    zipped_works_file_path.unlink()


images_path = data_dir / "images"
metadata_path = data_dir / "metadata"


print(f"Downloading images to {images_path.absolute()}...")
api_url = "https://api.wellcomecollection.org/catalogue/v2/works"
n_records = httpx.get(api_url).json()['totalResults']
n_images_found = 0
progress_bar = tqdm(load_records(works_file_path), total=n_records)
for record in progress_bar:
    if len(record['images']) > 0:
        for image in record['images']:
            # download the image
            iiif_info_url = httpx.get(
                f"https://api.wellcomecollection.org/catalogue/v2/images/{image['id']}"
            ).json()['thumbnail']['url']
            thumbnail_url = str(
                IIIFImageClient.init_from_url(iiif_info_url).size(height=460)
            )
            image_path = images_path / f"{image['id']}.jpg"
            image_path.touch(exist_ok=True)
            image_path.write_bytes(httpx.get(thumbnail_url).content)

            # save the record metadata
            record_path = metadata_path / f"{image['id']}.json"
            record_path.touch(exist_ok=True)
            with open(record_path, "w") as f:
                json.dump({
                    "image_path": str(image_path.absolute()),
                    **record
                }, f)

        n_images_found += len(record['images'])
        progress_bar.set_postfix({'images': n_images_found})

print("Deleting works dataset...")
works_file_path.unlink()
print(f"Done. {n_images_found} images downloaded.")
