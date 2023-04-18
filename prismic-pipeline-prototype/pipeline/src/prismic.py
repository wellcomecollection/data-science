from typing import Generator
import httpx
from .log import get_logger

log = get_logger()

api_url = "https://wellcomecollection.cdn.prismic.io/api/v2/"


def get_prismic_master_ref() -> str:
    response = httpx.get(api_url).json()
    return response["refs"][0]["ref"]


master_ref = get_prismic_master_ref()


def count_documents(document_type: str) -> int:
    response = httpx.get(
        api_url + "documents/search",
        params={"ref": master_ref,
                "q": f'[[at(document.type,"{document_type}")]]'},
    ).json()
    return response["total_results_size"]


def yield_documents(batch_size: int, limit: int, document_type: str) -> Generator[dict, None, None]:
    response = httpx.get(
        api_url + "documents/search",
        params={
            "ref": master_ref,
            "q": f'[[at(document.type,"{document_type}")]]',
            "pageSize": batch_size,
        },
    ).json()

    i = 0
    while True:
        for result in response["results"]:
            i += 1
            yield result
            if limit:
                if i >= limit:
                    return
        if response["next_page"] is None:
            break
        else:
            response = httpx.get(response["next_page"]).json()
