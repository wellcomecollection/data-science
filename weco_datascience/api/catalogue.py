import random
from . import api_url

catalogue_url = api_url / "works"
valid_query_args = set([
    "pageSize",
    "page",
    "query",
    "production.dates.from",
    "production.dates.to",
    "sortOrder",
    "sort",
    "license",
    "identifiers",
    "subjects.label",
    "genres.label",
    "language",
    "aggregations",
    "type",
    "workType",
    "items.locations.type",
    "items.locations.locationType",
    "include",
])


def get_work(query_id):
    response = (catalogue_url / query_id).get()
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        raise ValueError("Invalid ID")
    else:
        raise ValueError(f"{response.status_code} error", response)


def get_random_work():
    # only the first 10,000 works are available in api, so we limit the
    # chooseable pages to the first 1000
    random_page_number = random.randint(1, 1000)
    response = works_search(page=random_page_number)
    random_work = random.choice(response["results"])
    return random_work


def works_search(**kwargs):
    for key in kwargs:
        if key not in valid_query_args:
            raise ValueError(f"\"{key}\" is not a valid query parameter")
    response = catalogue_url.with_query(**kwargs).get()
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"{response.status_code} error")
