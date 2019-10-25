import json
import boto3

sqs_client = boto3.client('sqs')
s3 = boto3.client('s3')


def get_messages(queue_url):
    """Generates messages from an SQS queue.

    Note: this continues to generate messages until the queue is empty.
    Every message on the queue will be deleted.

    :param queue_url: URL of the SQS queue to drain.

    """
    while True:
        resp = sqs_client.receive_message(
            QueueUrl=queue_url,
            AttributeNames=['All'],
            MaxNumberOfMessages=10
        )

        try:
            yield from resp['Messages']
        except KeyError:
            return

        entries = [
            {'Id': msg['MessageId'], 'ReceiptHandle': msg['ReceiptHandle']}
            for msg in resp['Messages']
        ]

        resp = sqs_client.delete_message_batch(
            QueueUrl=queue_url, Entries=entries
        )

        if len(resp['Successful']) != len(entries):
            raise RuntimeError(
                "Failed to delete messages: entries={!r} resp={!r}"
                .format(entries, resp)
            )


def get_vhs_record_from_messages(messages):
    for message in messages:
        message_body = json.loads(message['Body'])
        vhs_record = json.loads(message_body['Message'])
        yield vhs_record


def get_calm_records_from_vhs_records(vhs_records):
    for record in vhs_records:
        s3_object = s3.get_object(
            Bucket=record['location']['namespace'],
            Key=record['location']['key'],
        )
        calm_record = json.loads(
            s3_object['Body'].read().decode('utf-8')
        )
        yield calm_record


def get_texts_from_calm_records(calm_records):
    for record in calm_records:
        try:
            description = record['Description'][0]
        except KeyError:
            description = ''
        try:
            title = record['Title'][0]
        except KeyError:
            title = ''
        yield (record['RecordID'][0], title, description)


def get_texts_from_queue(queue_url):
    messages = get_messages(queue_url)
    vhs_records = get_vhs_record_from_messages(messages)
    calm_records = get_calm_records_from_vhs_records(vhs_records)
    for record_id, title, description in get_texts_from_calm_records(calm_records):
        yield record_id, title, description
