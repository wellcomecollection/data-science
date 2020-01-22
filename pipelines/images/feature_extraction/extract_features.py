from io import BytesIO
import os

import click
from PIL import Image

from src.aws import get_object_from_s3
from src.feature_extraction import extract_features
from src.images import get_image

@click.command()
@click.option('--s3_bucket_name', '-n')
@click.option('--s3_object_key', '-k')
def main(s3_bucket_name, s3_object_key):
    image = get_image(s3_bucket_name, s3_object_key)
    image_features = extract_features(image)
    print(image_features)


if __name__ == "__main__":
    main()
