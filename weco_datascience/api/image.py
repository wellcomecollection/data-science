import random

from . import api_url

images_url = api_url / "images"
valid_query_args = set(
    ["query", "locations.license", "colors", "page", "pageSize"]
)


def get_image(query_id):
    response = (images_url / query_id).get()
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        raise ValueError("Invalid ID")
    else:
        raise ValueError(f"{response.status_code} error", response)


def get_random_image():
    # only the first 10,000 works are available in api, so we limit the
    # chooseable pages to the first 1000
    random_page_number = random.randint(1, 1000)
    response = image_search(page=random_page_number)
    random_image = random.choice(response["results"])
    return random_image


def image_search(**kwargs):
    for key in kwargs:
        if key not in valid_query_args:
            raise ValueError(f'"{key}" is not a valid query parameter')

    response = images_url.with_query(**kwargs).get()
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"{response.status_code} error")
