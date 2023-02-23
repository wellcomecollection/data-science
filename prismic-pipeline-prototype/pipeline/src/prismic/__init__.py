import httpx

api_url = "https://wellcomecollection.cdn.prismic.io/api/v2/"


def get_prismic_master_ref() -> str:
    response = httpx.get(api_url).json()
    return response["refs"][0]["ref"]


master_ref = get_prismic_master_ref()

from .articles import count_articles, yield_articles  # isort
from .exhibitions import count_exhibitions, yield_exhibitions  # isort
from .events import count_events, yield_events  # isort
from .people import count_people, yield_people  # isort

