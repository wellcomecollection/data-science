import boto3


def get_secret(secret_id, region_name="eu-west-1"):
    sts_client = boto3.client("sts")
    assumed_role = sts_client.assume_role(
        RoleArn="arn:aws:iam::760097843905:role/platform-developer",
        RoleSessionName="platform-dev-session",
    )
    session = boto3.session.Session(
        aws_access_key_id=assumed_role["Credentials"]["AccessKeyId"],
        aws_secret_access_key=assumed_role["Credentials"]["SecretAccessKey"],
        aws_session_token=assumed_role["Credentials"]["SessionToken"],
    )
    client = session.client(
        service_name="secretsmanager", region_name=region_name
    )
    response = client.get_secret_value(SecretId=secret_id)
    return response["SecretString"]
