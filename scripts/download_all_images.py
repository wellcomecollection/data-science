#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import json
from datetime import datetime
from io import BytesIO
from os import path

import click
import requests
from PIL import Image


def download_image(url, resolution):
    resized_url = url.replace('/300,/', f'/{resolution},/')
    content = requests.get(resized_url).content
    image = Image.open(BytesIO(content))
    return image


@click.command()
@click.option(
    '--json_path',
    help='Where the works json is saved (downloadable at https://developers.wellcomecollection.org/datasets)'
)
@click.option(
    '--image_path',
    help='Where the images should be saved'
)
@click.option(
    '--resolution',
    help='Width (in pixels) of the downloaded images',
    default=750
)
def main(json_path, image_path, resolution):
    """
    Download every image available on https://wellcomecollection.org/works 
    using the catalogue snapshots and IIIF API
    """
    start_time = datetime.now()
    n_lines = sum(1 for line in open(json_path))
    print(
        f'Started downloading images at {start_time}. {n_lines} records to parse.'
    )

    downloaded = 0
    with open(json_path) as f:
        for seen, line in enumerate(f):
            record = json.loads(line)
            if 'thumbnail' in record:
                save_path = path.join(image_path, record['id'] + '.jpg')
                url = record['thumbnail']['url']
                image = download_image(url, resolution)
                image.save(save_path)
                downloaded += 1

            if seen % 10 == 0:
                print(
                    f'Records seen: {seen}/{n_lines},  Images downloaded: {downloaded}',
                    end='\r'
                )


if __name__ == '__main__':
    main()
