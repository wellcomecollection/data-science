from io import BytesIO

import boto3


def get_assume_role_credentials(role_arn):
    sts = boto3.client('sts')
    credentials = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName='AssumeRoleSession1'
    )['Credentials']
    return credentials


def get_s3_client(credentials=None):
    if credentials:
        s3 = boto3.client(
            's3',
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken']
        )
    else:
        s3 = boto3.client('s3')
    return s3


def get_dynamo_client(credentials=None):
    if credentials:
        dynamodb = boto3.client(
            'dynamodb',
            region_name='eu-west-1',
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken']
        )
    else:
        dynamodb = boto3.client('dynamodb', region_name='eu-west-1')
    return dynamodb


def get_object_from_s3(s3, bucket, key):
    response = s3.get_object(
        Bucket=bucket,
        Key=key
    )
    data = response['Body'].read()
    return BytesIO(data)


def get_object_from_dynamo(dynamodb, table, key):
    response = dynamodb.get_item(
        TableName=table,
        Key={'id': {'S': key}}
    )
    try:
        dynamo_object = response['Item']
    except KeyError:
        raise ValueError(f'{key} is not a known miro id')
    return dynamo_object
