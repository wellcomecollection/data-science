
from weco_datascience.http import fetch_url_json
from weco_datascience.logging import get_logger

log = get_logger(__name__)


async def get_manifest_url_from_work_id(work_id):
    log.info(f"fetching the IIIF manifest for work ID: {work_id}")

    manifest_url = None
    response = await fetch_url_json(
        url=f"https://api.wellcomecollection.org/catalogue/v2/works/{work_id}",
        params={"include": "items"}
    )
    if response["object"].status == 200:
        try:
            for item in response["json"]["items"]:
                for location in item["locations"]:
                    if location["locationType"]["id"] == "iiif-presentation":
                        manifest_url = location["url"]
        except (KeyError, ValueError):
            raise ValueError(
                f"Couldn't locate a IIIF manifest for work ID: {work_id}"
            )
    else:
        raise ValueError(f"It doesn't look like {work_id} is a valid work ID")

    if manifest_url:
        return manifest_url
    else:
        raise ValueError(
            f"Couldn't locate a IIIF manifest for work ID: {work_id}"
        )
