import os
from io import BytesIO

import boto3

s3 = boto3.client('s3')


def get_object_from_s3(object_key):
    response = s3.get_object(
        Bucket='model-core-data',
        Key=object_key
    )
    data = response['Body'].read()
    return BytesIO(data)
