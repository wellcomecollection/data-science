import httpx

elastic_config = httpx.get(
    "https://api.wellcomecollection.org/catalogue/v2/_elasticConfig"
).json()

images_index = elastic_config["imagesIndex"]
works_index = elastic_config["worksIndex"]

pipeline_date = "-".join(works_index.split("-")[-3:])
