import os

import boto3
from botocore.exceptions import ClientError

from .logging import get_logger

log = get_logger(__name__)


def get_s3_client():
    try:
        s3_client = boto3.client("s3")
    except ClientError as e:
        log.error(f"Failed to create s3 client: {e}")
        raise e
    return s3_client


def download_object_from_s3(bucket_name, object_key, file_name=None):
    s3_client = get_s3_client()
    s3_client.download_file(
        Bucket=bucket_name,
        Key=object_key,
        Filename=(file_name or os.path.basename(object_key)),
    )


def get_session(*, role_arn):
    """
    Returns a boto3 Session authenticated with the current role ARN.
    """
    sts_client = boto3.client("sts")
    assumed_role_object = sts_client.assume_role(
        RoleArn=role_arn, RoleSessionName="AssumeRoleSession1"
    )
    credentials = assumed_role_object["Credentials"]
    return boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )


def get_secret_string(session, *, secret_id):
    """
    Look up the value of a SecretString in Secrets Manager.
    """
    secrets = session.client("secretsmanager")
    return secrets.get_secret_value(SecretId=secret_id)["SecretString"]
