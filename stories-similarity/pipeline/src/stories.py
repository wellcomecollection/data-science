from typing import Generator, Optional

import httpx

api_url = "https://wellcomecollection.cdn.prismic.io/api/v2/"


def get_prismic_master_ref() -> str:
    response = httpx.get(api_url).json()
    return response["refs"][0]["ref"]


master_ref = get_prismic_master_ref()


def count_stories() -> int:
    response = httpx.get(
        api_url + "documents/search",
        params={"ref": master_ref, "q": '[[at(document.type,"articles")]]'},
    ).json()
    return response["total_results_size"]


def yield_stories(
    batch_size: int, limit: Optional[int] = None
) -> Generator[dict, None, None]:
    response = httpx.get(
        api_url + "documents/search",
        params={
            "ref": master_ref,
            "q": '[[at(document.type,"articles")]]',
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
