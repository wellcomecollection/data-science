import pickle

import boto3
import numpy as np

from .aws import get_object_from_s3
from boto3.dynamodb.conditions import Key

# Assume role and get credentials to read from miro dynamo table
sts = boto3.client('sts')
credentials = sts.assume_role(
    RoleArn='arn:aws:iam::760097843905:role/sourcedata-miro-assumable_read_role',
    RoleSessionName='AssumeRoleSession1'
)['Credentials']

dynamodb = boto3.resource(
    'dynamodb',
    region_name='eu-west-1',
    aws_access_key_id=credentials['AccessKeyId'],
    aws_secret_access_key=credentials['SecretAccessKey'],
    aws_session_token=credentials['SessionToken']
)

# Find miro_ids which are cleared for the catalogue api, and only use these
# as valid results in the palette api
table = dynamodb.Table('vhs-sourcedata-miro')
filter_expression = Key('isClearedForCatalogueAPI').eq(True)
response = table.scan(
    ProjectionExpression='id',
    FilterExpression=filter_expression
)
miro_ids_cleared_for_catalogue_api = set(
    item['id'] for item in response['Items']
)
while 'LastEvaluatedKey' in response:
    response = table.scan(
        ProjectionExpression='id',
        FilterExpression=filter_expression,
        ExclusiveStartKey=response['LastEvaluatedKey']
    )
    miro_ids_cleared_for_catalogue_api.update(set(
        item['id'] for item in response['Items']
    ))

miro_ids = np.load(get_object_from_s3('palette/image_ids.npy'))

catalogue_id_to_miro_id = pickle.load(
    get_object_from_s3('palette/catalogue_id_to_miro_id.pkl')
)

valid_catalogue_ids = set(
    catalogue_id
    for catalogue_id, miro_id in catalogue_id_to_miro_id.items()
    if miro_id in miro_ids_cleared_for_catalogue_api
)


def miro_id_to_miro_uri(miro_id):
    return (
        "https://iiif.wellcomecollection.org/"
        f"image/{miro_id}.jpg/full/960,/0/default.jpg"
    )


def miro_id_to_catalogue_uri(miro_id):
    catalogue_id = all_miro_id_to_catalogue_id[miro_id]
    return 'https://wellcomecollection.org/works/' + catalogue_id


def miro_id_to_identifiers(miro_id):
    return {
        'miro_id': miro_id,
        'catalogue_id': all_miro_id_to_catalogue_id[miro_id],
        'miro_uri': miro_id_to_miro_uri(miro_id),
        'catalogue_uri': miro_id_to_catalogue_uri(miro_id)
    }


def filter_invalid_ids(neighbour_ids, n):
    valid_ids = [
        miro_id for miro_id in neighbour_ids
        if miro_id in miro_ids_cleared_for_catalogue_api
    ]
    return valid_ids[:n]
