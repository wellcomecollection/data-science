import os

import click
import numpy as np
from PIL import Image

from src.feature_extraction import extract_image_features
from src.file_utils import (file_name_from_path, file_names_in_dir,
                            paths_from_dir)
from src.images import get_images, load_images, build_image_grid
from src.spaces import get_assignments, make_grid, squash_embeddings

IMAGE_DIR = os.path.abspath('data/raw/images/')
FEATURE_DIR = os.path.abspath('data/processed/features/')


@click.command()
@click.option('--query')
@click.option('--side_length', default=20)
def main(query, side_length):
    n_images = side_length ** 2

    # get images from api.wellcomecollection.org, save them in IMAGE_DIR
    # get_images(query, n_images, IMAGE_DIR)

    # extract features from images, store them in FEATURE_DIR
    extract_image_features(feature_dir=FEATURE_DIR, image_dir=IMAGE_DIR)

    # load embeddings and images
    images = load_images(IMAGE_DIR)
    image_ids = list(images.keys())
    feature_paths = [
        os.path.join(FEATURE_DIR, image_id) + '.npy'
        for image_id in image_ids
    ]

    embeddings = np.stack([np.load(path) for path in feature_paths])
    embeddings_2d = squash_embeddings(embeddings)

    # get best assignments for square grid
    assignments = get_assignments(embeddings_2d, side_length)
    image = build_image_grid(images, assignments)
    image.save(f'data/processed/{query}.jpg')


if __name__ == "__main__":
    main()
