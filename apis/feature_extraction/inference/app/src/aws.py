import os
from io import BytesIO

import boto3


def get_s3_client(session=None, profile_name=None):
    if profile_name:
        session = boto3.session.Session(
            profile_name=profile_name, region_name='eu-west-1'
        )
        s3_client = session.client('s3')
    elif session:
        s3_client = session.client('s3')
    else:
        raise ValueError(
            'Need an existing session or a profile name with which to create one'
        )
    return s3_client


def get_object_from_s3(object_key, bucket_name, session=None, profile_name=None):
    s3_client = get_s3_client(session, profile_name)
    response = s3_client.get_object(
        Bucket=bucket_name,
        Key=object_key
    )
    return response['Body'].read()


def download_object_from_s3(object_key, bucket_name, session=None, profile_name=None):
    s3_client = get_s3_client(session, profile_name)
    s3_client.download_file(
        Bucket=bucket_name,
        Key=object_key,
        Filename=os.path.basename(object_key)
    )
