import json
import os
import pickle
from io import BytesIO

import boto3

DATA_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'miro_id_to_catalogue_id.pkl'
)

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, 'rb') as f:
        identifiers = pickle.load(f)
else:
    identifiers = {}

already_fetched = set(identifiers.keys())

sts = boto3.client('sts')
credentials = sts.assume_role(
    RoleArn='arn:aws:iam::760097843905:role/miro-migration-assumable_read_role',
    RoleSessionName='AssumeRoleSession1'
)['Credentials']

s3 = boto3.client(
    's3',
    aws_access_key_id=credentials['AccessKeyId'],
    aws_secret_access_key=credentials['SecretAccessKey'],
    aws_session_token=credentials['SessionToken']
)

dynamodb = boto3.resource(
    'dynamodb',
    region_name='eu-west-1',
    aws_access_key_id=credentials['AccessKeyId'],
    aws_secret_access_key=credentials['SecretAccessKey'],
    aws_session_token=credentials['SessionToken']
)

table = dynamodb.Table('vhs-miro-migration')
response = table.scan()
data = response['Items']

while 'LastEvaluatedKey' in response:
    response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
    data.extend(response['Items'])


def get_object_from_s3(bucket_name, object_key):
    response = s3.get_object(
        Bucket=bucket_name,
        Key=object_key
    )
    data = response['Body'].read()
    return BytesIO(data)


def get_identifiers(vhs_entry):
    object_key = vhs_entry['location']['key']
    s3_obj = get_object_from_s3(
        bucket_name='wellcomecollection-vhs-miro-migration',
        object_key=object_key
    )
    json_obj = json.load(s3_obj)
    miro_id = json_obj['id']
    catalogue_id = json_obj['catalogue_entry_id']
    return miro_id, catalogue_id


for i, vhs_entry in enumerate(data):
    if vhs_entry['id'] in already_fetched:
        pass
    else:
        miro_id, catalogue_id = get_identifiers(vhs_entry)
        identifiers[miro_id] = catalogue_id

    if i % 500 == 0:
        print(i, len(identifiers))
        with open(DATA_PATH, 'wb') as f:
            pickle.dump(identifiers, f)

with open(DATA_PATH, 'wb') as f:
    pickle.dump(identifiers, f)
