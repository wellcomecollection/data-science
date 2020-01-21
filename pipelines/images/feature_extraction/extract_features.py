from io import BytesIO

import click
from PIL import Image

from src.aws import get_object_from_s3
from src.feature_extraction import extract_features


@click.command()
@click.option('--s3_bucket_name', '-n')
@click.option('--s3_object_key', '-k')
def main(s3_bucket_name, s3_object_key):
    image_bytes = get_object_from_s3(
        bucket_name=s3_bucket_name,
        object_key=s3_object_key,
        profile_name='platform-dev'
    )
    image = Image.open(BytesIO(image_bytes))
    image_features = extract_features(image)
    print(image_features)


if __name__ == "__main__":
    main()
