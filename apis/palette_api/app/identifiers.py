import pickle

import pandas as pd

from .aws import get_object_from_s3

df = pd.DataFrame(pickle.load(get_object_from_s3('palette/identifiers.pkl'))).T

valid_catalogue_ids = (
    df[df['is_cleared_for_catalogue_api']]
    [['miro_catalogue_id', 'sierra_catalogue_id']]
    .values.reshape(-1)
)

miro_ids_in_nmslib_order = df['palette_index'].sort_values().index.values

miro_ids_cleared_for_catalogue_api = set(
    df[df['is_cleared_for_catalogue_api']].index.values
)

catalogue_id_to_miro_id = {
    **{v: k for k, v in df['sierra_catalogue_id'].items()},
    **{v: k for k, v in df['miro_catalogue_id'].items()}
}

index_lookup = df['palette_index'].to_dict()


def miro_id_to_miro_uri(miro_id):
    return (
        "https://iiif.wellcomecollection.org/"
        f"image/{miro_id}.jpg/full/960,/0/default.jpg"
    )


def miro_id_to_catalogue_uri(miro_id):
    catalogue_id = df['sierra_catalogue_id'][miro_id]
    return 'https://wellcomecollection.org/works/' + catalogue_id


def miro_id_to_identifiers(miro_id):
    return {
        'miro_id': miro_id,
        'catalogue_id': df['sierra_catalogue_id'][miro_id],
        'miro_uri': miro_id_to_miro_uri(miro_id),
        'catalogue_uri': miro_id_to_catalogue_uri(miro_id)
    }


def filter_invalid_ids(neighbour_ids, n):
    valid_ids = [
        miro_id for miro_id in neighbour_ids
        if miro_id in miro_ids_cleared_for_catalogue_api
    ]
    return valid_ids[:n]
