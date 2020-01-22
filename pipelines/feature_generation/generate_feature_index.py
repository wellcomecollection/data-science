import json
import os
import pickle
from io import BytesIO
from os import mkdir
from os.path import exists, join, realpath, split

import click
import numpy as np
import pandas as pd
from halo import Halo
from PIL import Image
from tqdm import tqdm

from src.ann import build_ann_index, save_ann_index
from src.aws import get_keys_and_ids, get_object_from_s3, put_object_to_s3
from src.feature_extraction import extract_and_save_image_features
from src.utils import (get_miro_to_catalogue_id_lookup, save_catalogue_ids,
                       save_catalogue_miro_lookup)


@click.command()
@click.option('--works_json_path')
@click.option('--feature_vector_dir')
def main(works_json_path, feature_vector_dir):
    _, image_ids, image_ids_to_keys = get_keys_and_ids(
        profile_name='platform-dev',
        bucket_name='wellcomecollection-miro-images-public',
    )

    _, feature_ids, _ = get_keys_and_ids(
        profile_name='data-dev',
        bucket_name='miro-images-feature-vectors',
        prefix='feature_vectors/'
    )

    missing_ids = list(set(image_ids) - set(feature_ids))
    if len(missing_ids) > 0:
        print('\nExtracting missing features from images')
        for miro_id in missing_ids:
            image_key = image_ids_to_keys[miro_id]
            extract_and_save_image_features(miro_id, image_key)

    miro_id_to_catalogue_id = get_miro_to_catalogue_id_lookup(works_json_path)

    data = {}
    print('\nLoading features')
    for miro_id in tqdm(feature_ids):
        with open(join(feature_vector_dir, miro_id)) as f:
            feature_vector = np.fromfile(f, dtype=np.float32)

        catalogue_id = (
            miro_id_to_catalogue_id[miro_id]
            if miro_id in miro_id_to_catalogue_id
            else None
        )

        data[miro_id] = {
            'catalogue_id': catalogue_id,
            'is_cleared_for_catalogue_api': bool(catalogue_id),
            'feature_vector': feature_vector
        }

    df = pd.DataFrame.from_dict(data).T
    cleared_for_api = df[df['is_cleared_for_catalogue_api']]
    feature_vectors = np.stack(cleared_for_api['feature_vector'].values)
    catalogue_ids = cleared_for_api['catalogue_id'].values
    catalogue_id_to_miro_id = {
        k: v for v, k in miro_id_to_catalogue_id.items()
    }

    ann_index = build_ann_index(feature_vectors)

    save_ann_index(ann_index)
    save_catalogue_ids(catalogue_ids)
    save_catalogue_miro_lookup(catalogue_id_to_miro_id)
    print('Done!')


if __name__ == "__main__":
    main()
