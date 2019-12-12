import json

import numpy as np
import requests

from .aws import get_s3_client, get_object_from_s3, get_assume_role_credentials, get_object_from_dynamo

data_credentials = get_assume_role_credentials(
    'arn:aws:iam::964279923020:role/data-developer'
)
data_s3 = get_s3_client(data_credentials)
palette_ordered_ids = np.load(
    get_object_from_s3(data_s3, 'model-core-data', 'palette/image_ids.npy')
)
feature_ordered_ids = np.load(
    get_object_from_s3(data_s3, 'model-core-data', 'image_pathways/ids.npy')
)


def get_catalogue_id_miro(platform_dynamo, platform_s3, key):
    dynamo_response = get_object_from_dynamo(
        platform_dynamo, 'vhs-miro-migration', key
    )
    miro_sourcedata_response = json.load(get_object_from_s3(
        s3=platform_s3,
        bucket='wellcomecollection-vhs-miro-migration',
        key=dynamo_response['location']['M']['key']['S']
    ))
    try:
        catalogue_id_miro = miro_sourcedata_response['catalogue_entry_id']
    except KeyError:
        catalogue_id_miro = None
    return catalogue_id_miro


def choose_correct_catalogue_id(miro_id, results):
    sierra_catalogue_id = None
    for work in results:
        try:
            if miro_id in work['thumbnail']['url']:
                sierra_catalogue_id = work['id']
        except KeyError:
            pass
    return sierra_catalogue_id


def get_catalogue_id_sierra(miro_id):
    try:
        base_url = 'https://api.wellcomecollection.org/catalogue/v2/works?query='
        results = requests.get(base_url + miro_id).json()['results']
        sierra_catalogue_id = choose_correct_catalogue_id(miro_id, results)
    except:
        sierra_catalogue_id = None
    return sierra_catalogue_id


def get_palette_index(miro_id):
    if miro_id in palette_ordered_ids:
        palette_index = int(np.where(palette_ordered_ids == miro_id)[0][0])
    else:
        palette_index = None
    return palette_index


def get_feature_index(miro_id):
    if miro_id in feature_ordered_ids:
        feature_index = int(np.where(feature_ordered_ids == miro_id)[0][0])
    else:
        feature_index = None
    return feature_index
