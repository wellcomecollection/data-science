import math
import csv
from pathlib import Path

import httpx
import typer
from weco_datascience.logging import get_logger

log = get_logger(__name__)


def fetch_work_ids(query, n):
    log.info(f"fetching {n} works from api.wellcomecollection.org")
    page, work_ids = 1, []
    while len(work_ids) < n:
        response = httpx.get(
            url="https://api.wellcomecollection.org/catalogue/v2/works",
            params={
                "query": query,
                "workType": "a",
                "items.locations.type": "DigitalLocation",
                "pageSize": 100,
                "page": page,
            }
        )
        work_ids.extend([
            work["id"]
            for work in response.json()["results"]
        ])
        page += 1
    return work_ids[:n]


def save_work_ids(work_ids, output_path):
    if not output_path.exists():
        if not output_path.parent.exists():
            log.info(f"creating {output_path.parent} directory")
            output_path.mkdir(parents=True)
        log.info(f"creating {output_path} file")
        output_path.touch()

    with open(output_path, 'w') as f:
        log.info(f"writing {len(work_ids)} work ids to {output_path}")
        csv.writer(f).writerow(work_ids)
        

def main(query: str = "", n: int = 100, output_path: Path = Path("/data/work_ids.csv"), verbose: bool = False):
    work_ids = fetch_work_ids(query, n)
    save_work_ids(work_ids, output_path)
    if verbose:
        log.info(" ".join(work_ids))


if __name__ == "__main__":
    typer.run(main)
