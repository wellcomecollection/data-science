import json
from pathlib import Path
from typing import Tuple


def transform_data(data, type):
    if type == "articles":
        return transform_article(data)
    elif type == "exhibitions":
        return transform_exhibition(data)
    elif type == "events":
        return transform_event(data)
    else:
        raise ValueError(f"Unknown type: {type}")


contributor_data_dir = Path("/data/prismic/people")
contributor_data_dir.mkdir(parents=True, exist_ok=True)
id_to_name = {}
for file in contributor_data_dir.iterdir():
    data_path = contributor_data_dir / file
    data = json.loads(data_path.read_text(encoding="utf-8"))
    id_to_name[data["id"]] = data["data"]["name"]


def transform_article(data: dict) -> Tuple[str, dict]:
    standfirst = []
    body = []
    for slice in data["data"]["body"]:
        if slice["slice_type"] == "standfirst":
            for paragraph in slice["primary"]["text"]:
                standfirst.append(paragraph["text"])
        if slice["slice_type"] == "text":
            for paragraph in slice["primary"]["text"]:
                body.append(paragraph["text"])

    title = data["data"]["title"][0]["text"]
    published = data["first_publication_date"]
    promo_image = data["data"]["promo"][0]["primary"]["image"]["url"]
    try:
        promo_caption = data["data"]["promo"][0]["primary"]["caption"][0]["text"]
    except (KeyError, IndexError):
        promo_caption = None

    contributors = []
    for contributor in data["data"]["contributors"]:
        try:
            contributors += [id_to_name[contributor["contributor"]["id"]]]
        except KeyError:
            pass

    document = {
        "type": "article",
        "title": title,
        "standfirst": standfirst,
        "body": body,
        "published": published,
        "promo_image": promo_image,
        "promo_caption": promo_caption,
        "contributors": contributors,
    }
    return data["id"], document


def transform_exhibition(data: dict) -> Tuple[str, dict]:
    body = []
    for slice in data["data"]["body"]:
        if slice["slice_type"] == "text":
            for paragraph in slice["primary"]["text"]:
                body.append(paragraph["text"])
    title = data["data"]["title"][0]["text"]
    published = data["first_publication_date"]
    starts = data["data"]["start"]
    ends = data["data"]["end"]
    contributors = []
    for contributor in data["data"]["contributors"]:
        try:
            contributors += [id_to_name[contributor["contributor"]["id"]]]
        except KeyError:
            pass
    try:
        promo_image = data["data"]["promo"][0]["primary"]["image"]["url"]
    except (KeyError, IndexError):
        promo_image = None
    try:
        promo_caption = data["data"]["promo"][0]["primary"]["caption"][0]["text"]
    except (KeyError, IndexError):
        promo_caption = None
    document = {
        "type": "exhibition",
        "title": title,
        "body": body,
        "published": published,
        "starts": starts,
        "ends": ends,
        "promo_image": promo_image,
        "promo_caption": promo_caption,
        "contributors": contributors
    }
    return data["id"], document


def transform_event(data: dict) -> Tuple[str, dict]:
    body = []
    for slice in data["data"]["body"]:
        if slice["slice_type"] == "text":
            for paragraph in slice["primary"]["text"]:
                body.append(paragraph["text"])
    title = data["data"]["title"][0]["text"]
    published = data["first_publication_date"]
    starts, ends = [], []
    for date in data["data"]["times"]:
        starts.append(date["startDateTime"])
        ends.append(date["endDateTime"])
    try:
        promo_image = data["data"]["promo"][0]["primary"]["image"]["url"]
    except (KeyError, IndexError):
        promo_image = None
    try:
        promo_caption = data["data"]["promo"][0]["primary"]["caption"][0]["text"]
    except (KeyError, IndexError):
        promo_caption = None
    contributors = []
    for contributor in data["data"]["contributors"]:
        try:
            contributors += [id_to_name[contributor["contributor"]["id"]]]
        except KeyError:
            pass
    document = {
        "type": "event",
        "title": title,
        "body": body,
        "published": published,
        "starts": starts,
        "ends": ends,
        "promo_image": promo_image,
        "promo_caption": promo_caption,
        "contributors": contributors
    }
    return data["id"], document
