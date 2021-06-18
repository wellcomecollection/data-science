import json
from wasabi import Printer

from tqdm import tqdm
import httpx
from src.image import get_image_url_from_iiif_url

msg = Printer()

with msg.loading("fetching general detail from the API"):
    first_response = httpx.get(
        "https://api.wellcomecollection.org/catalogue/v2/images",
        params={
            "pageSize": 100
        }
    ).json()
msg.good("fetched general detail from the API")

n_pages_to_crawl = 10  # first_response['totalPages']
msg.info(
    f"fetching {n_pages_to_crawl} of "
    f"{first_response['totalPages']} available pages"
)

image_urls = []
for i in tqdm(range(n_pages_to_crawl)):
    response = httpx.get(
        "https://api.wellcomecollection.org/catalogue/v2/images",
        params={
            "pageSize": 100,
            "page": i + 1
        }
    ).json()

    page_image_urls = [
        get_image_url_from_iiif_url(result['thumbnail']['url'])
        for result in response['results']
    ]
    image_urls.extend(page_image_urls)

data_path = "/data/image_urls.json"
msg.good(f"fetched {len(image_urls)} image URLs from the API")

with msg.loading(f"writing data to {data_path}"):
    with open(data_path, 'w') as f:
        json.dump(image_urls, f)
msg.good(f"wrote data to {data_path}")
