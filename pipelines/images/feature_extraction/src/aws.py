from os.path import basename, splitext

import boto3
from halo import Halo


def get_id_from_key(key):
    return splitext(basename(key))[0]


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


def put_object_to_s3(binary_object, key, bucket_name, session=None, profile_name=None):
    s3_client = get_s3_client(session, profile_name)
    s3_client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=binary_object
    )


def list_keys_in_bucket(profile_name, bucket_name, prefix=''):
    spinner = Halo(f'listing keys in {bucket_name}').start()

    s3_client = get_s3_client(profile_name=profile_name)

    keys = []
    kwargs = {'Bucket': bucket_name, 'Prefix': prefix}
    while True:
        resp = s3_client.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            keys.append(obj['Key'])
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break

    spinner.succeed(f'{len(keys)} keys in {bucket_name}')
    return keys


def get_keys_and_ids(profile_name, bucket_name, prefix=''):
    keys = list_keys_in_bucket(profile_name, bucket_name, prefix)
    ids_to_keys = {get_id_from_key(key): key for key in keys}
    ids, _ = list(zip(*ids_to_keys.items()))
    return keys, ids, ids_to_keys
