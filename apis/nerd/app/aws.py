import os
import json

import boto3
import numpy as np
import torch


if 'DEVELOPMENT' in os.environ:
    sts = boto3.client('sts')
    assumed_role_object = sts.assume_role(
        RoleArn='arn:aws:iam::964279923020:role/data-developer',
        RoleSessionName='AssumeRoleSession1'
    )
    credentials = assumed_role_object['Credentials']

    s3 = boto3.client(
        's3',
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken']
    )

    dynamo = boto3.client(
        'dynamodb',
        'eu-west-1',
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken']
    )

else:
    s3 = boto3.client('s3')
    dynamo = boto3.client('dynamodb', 'eu-west-1')


def get_object_from_s3(object_name):
    response = s3.get_object(
        Bucket='model-core-data',
        Key='nerd/' + object_name
    )
    return response['Body'].read()


def get_wikidata_embedding(wikidata_id):
    response = dynamo.get_item(
        TableName='wikidata-biggraph-embeddings',
        Key={'wikidata_id': {'S': str(wikidata_id)}}
    )

    if 'Item' not in response:
        raise ValueError(
            '"{}" is not a recognised wikidata_id!'.format(wikidata_id)
        )

    array_as_bytes = response['Item']['embedding']['B']

    unpacked_embedding = torch.Tensor(np.frombuffer(
        array_as_bytes, dtype=np.float32
    ))

    return unpacked_embedding
