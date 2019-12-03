import json
import os
import pickle
from io import BytesIO

import boto3
import numpy as np
import requests

sts = boto3.client('sts')

platform_credentials = sts.assume_role(
    RoleArn='arn:aws:iam::760097843905:role/platform-developer',
    RoleSessionName='AssumeRoleSession1'
)['Credentials']

s3_platform = boto3.client(
    's3',
    aws_access_key_id=platform_credentials['AccessKeyId'],
    aws_secret_access_key=platform_credentials['SecretAccessKey'],
    aws_session_token=platform_credentials['SessionToken']
)

dynamodb = boto3.resource(
    'dynamodb',
    region_name='eu-west-1',
    aws_access_key_id=platform_credentials['AccessKeyId'],
    aws_secret_access_key=platform_credentials['SecretAccessKey'],
    aws_session_token=platform_credentials['SessionToken']
)

migration_table = dynamodb.Table('vhs-miro-migration')
sourcedata_table = dynamodb.Table('vhs-sourcedata-miro')

data_credentials = sts.assume_role(
    RoleArn='arn:aws:iam::964279923020:role/data-developer',
    RoleSessionName='AssumeRoleSession1'
)['Credentials']

s3_data = boto3.client(
    's3',
    aws_access_key_id=data_credentials['AccessKeyId'],
    aws_secret_access_key=data_credentials['SecretAccessKey'],
    aws_session_token=data_credentials['SessionToken']
)


def get_object_from_s3(s3, bucket_name, object_key):
    response = s3.get_object(
        Bucket=bucket_name,
        Key=object_key
    )
    data = response['Body'].read()
    return BytesIO(data)


def get_miro_catalogue_id(miro_id):
    try:
        response = migration_table.get_item(Key={'id': miro_id})
        object_key = response['Item']['location']['key']
        s3_response = json.load(get_object_from_s3(
            s3=s3_platform,
            bucket_name='wellcomecollection-vhs-miro-migration',
            object_key=object_key
        ))
        miro_catalogue_id = s3_response['catalogue_entry_id']
    except:
        miro_catalogue_id = None
    return miro_catalogue_id


def choose_correct_catalogue_id(miro_id, results):
    sierra_catalogue_id = None
    for work in results:
        try:
            if miro_id in work['thumbnail']['url']:
                sierra_catalogue_id = work['id']
        except KeyError:
            pass
    return sierra_catalogue_id


def get_sierra_catalogue_id(miro_id):
    try:
        base_url = 'https://api.wellcomecollection.org/catalogue/v2/works?query='
        results = requests.get(base_url + miro_id).json()['results']
        sierra_catalogue_id = choose_correct_catalogue_id(miro_id, results)
    except:
        sierra_catalogue_id = None
    return sierra_catalogue_id


def is_cleared_for_catalogue_api(miro_id):
    try:
        response = sourcedata_table.get_item(Key={'id': miro_id})
        is_clear = response['Item']['isClearedForCatalogueAPI']
    except:
        is_clear = False
    return is_clear


palette_ordered_miro_ids = np.load(get_object_from_s3(
    s3_data, 'model-core-data', 'palette/image_ids.npy'
))

DATA_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'identifiers.pkl'
)

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, 'rb') as f:
        identifiers = pickle.load(f)
else:
    identifiers = {}

start_point = 0
for i, miro_id in enumerate(palette_ordered_miro_ids[start_point:]):
    print(i+start_point)
    process = True
    if miro_id in identifiers:
        process = False
        if identifiers[miro_id]['miro_catalogue_id'] == None:
            process = True

    if process:
        identifiers[miro_id] = {
            'miro_catalogue_id': get_miro_catalogue_id(miro_id),
            'sierra_catalogue_id': get_sierra_catalogue_id(miro_id),
            'is_cleared_for_catalogue_api': is_cleared_for_catalogue_api(miro_id),
            'palette_index': i+start_point
        }
        if i % 500 == 0:
            with open(DATA_PATH, 'wb') as f:
                pickle.dump(identifiers, f)
    else:
        pass

with open(DATA_PATH, 'wb') as f:
    pickle.dump(identifiers, f)
