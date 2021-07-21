import os
from io import BytesIO
from os.path import join

import numpy as np
import requests
from halo import Halo
from PIL import Image
from tqdm import tqdm

from .file_utils import file_name_from_path, paths_from_dir


def save_image(thumbnail_url, path):
    response = requests.get(thumbnail_url)
    image = Image.open(BytesIO(response.content))
    image.save(path)


def get_thumbnail_urls(query, n_images):
    spinner = Halo('getting image thumbnail urls from API').start()
    thumbnail_urls, image_ids = {}, []

    query_urls = [
        'https://api.wellcomecollection.org/catalogue/v2/works'
        f'?pageSize=100&query={query}&workType=q,k&page={page + 1}'
        for page in range(int(n_images / 90) + 1)
    ]

    for query_url in query_urls:
        response_json = requests.get(query_url).json()
        for result in response_json['results']:
            if 'thumbnail' in result and result['id'] not in thumbnail_urls:
                thumbnail_urls[result['id']] = result['thumbnail']['url']
                image_ids.append(result['id'])

    n_thumbnail_urls = {
        image_id: thumbnail_urls[image_id]
        for image_id in image_ids[:n_images]
    }
    spinner.succeed()
    return n_thumbnail_urls


def get_images(query, n_images, image_dir):
    thumbnail_urls = get_thumbnail_urls(query, n_images)
    loop = tqdm(thumbnail_urls.items())
    for image_id, thumbnail_url in loop:
        loop.set_description(f'saving image: {image_id}')
        path = f'{os.path.join(image_dir, image_id)}.jpg'
        save_image(thumbnail_url, path)


def load_images(image_path):
    paths = paths_from_dir(image_path)
    images = {
        file_name_from_path(path): Image.open(path) for path in paths
    }
    return images


def build_image_grid(images, assignments):
    ix_to_image_id = dict(enumerate(images.keys()))

    # construct square grid of images
    id_grid = [[ix_to_image_id[ix] for ix in row] for row in assignments]

    # build output grid of images
    image = Image.fromarray(
        np.concatenate([
            np.concatenate(
                [np.array(images[image_id].resize((50, 50)))
                 for image_id in row],
                axis=1
            )
            for row in id_grid
        ])
    )

    return image
