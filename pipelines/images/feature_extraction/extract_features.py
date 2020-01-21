import json
import os
import pickle
from io import BytesIO
from os import mkdir
from os.path import exists, join, realpath, split

import click
import numpy as np
from halo import Halo
from PIL import Image
from tqdm import tqdm
import pandas as pd

from src.aws import get_object_from_s3, put_object_to_s3
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
