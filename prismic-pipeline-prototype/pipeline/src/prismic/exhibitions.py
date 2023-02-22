from typing import Generator

import httpx

from . import api_url, master_ref


def count_exhibitions() -> int:
    response = httpx.get(
        api_url + "documents/search",
        params={"ref": master_ref, "q": '[[at(document.type,"exhibitions")]]'},
    ).json()
    return response["total_results_size"]


def yield_exhibitions(
    batch_size: int, limit: int
) -> Generator[dict, None, None]:
    response = httpx.get(
        api_url + "documents/search",
        params={
            "ref": master_ref,
            "q": '[[at(document.type,"exhibitions")]]',
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
