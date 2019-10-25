import json
import os
from io import BytesIO

import boto3
import numpy as np

if 'DEVELOPMENT' in os.environ:
    s3 = boto3.client('s3')
else:
    sts = boto3.client('sts')
    credentials = sts.assume_role(
        RoleArn='arn:aws:iam::964279923020:role/data-developer',
        RoleSessionName='AssumeRoleSession1'
    )['Credentials']

    s3 = boto3.client(
        's3',
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken']
    )


def get_object_from_s3(object_key):
    response = s3.get_object(
        Bucket='model-core-data',
        Key=object_key
    )
    data = response['Body'].read()
    return BytesIO(data)


def download_object_from_s3(object_key):
    s3.download_file(
        Bucket='model-core-data',
        Key=object_key,
        Filename=os.path.basename(object_key)
    )
