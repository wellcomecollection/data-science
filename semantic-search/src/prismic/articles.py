from typing import Generator

import httpx

from . import api_url, master_ref


def count_articles() -> int:
    response = httpx.get(
        api_url + "documents/search",
        params={"ref": master_ref, "q": '[[at(document.type,"articles")]]'},
    ).json()
    return response["total_results_size"]


def yield_articles(batch_size: int, limit: int = 1000) -> Generator[dict, None, None]:
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
