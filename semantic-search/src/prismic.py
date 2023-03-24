from typing import Generator

import httpx

from .prismic import api_url, master_ref


api_url = "https://wellcomecollection.cdn.prismic.io/api/v2"


def get_master_ref() -> str:
    response = httpx.get(api_url).json()
    return [
        ref["ref"]
        for ref in response["refs"]
        if ref["isMasterRef"]
    ][0]


master_ref = get_master_ref()


def count_documents() -> int:
    response = httpx.get(
        api_url + "documents/search",
        params={"ref": master_ref},
    ).json()
    return response["total_results_size"]


def get_document_types() -> list:
    response = httpx.get(
        api_url,
        params={"ref": master_ref},
    ).json()
    return response["types"].keys()


def yield_documents(batch_size: int) -> Generator[dict, None, None]:
    for type in get_document_types():
        response = httpx.get(
            api_url + "documents/search",
            params={
                "ref": master_ref,
                "q": f"[[at(document.type, \"{type}\")]]",
                "pageSize": batch_size,
            },
        ).json()

        i = 0
        while True:
            for result in response["results"]:
                i += 1
                yield result
            if response["next_page"] is None:
                break
            else:
                response = httpx.get(response["next_page"]).json()
