import json
import boto3
import click
from tqdm import tqdm
from nerd.poll_queue import get_texts_from_queue
from nerd.extract import extract_entities


@click.command()
@click.option(
    "--source_queue_name",
    prompt="Source queue name",
    default="inference-entity-extraction",
    help="name of the queue whose messages we want to process"
)
@click.option(
    "--dest_bucket_name",
    prompt="Destination bucket name",
    default="wellcomecollection-inference-calm",
    help="the name of the s3 bucket where we'll post results"
)
def process_queue(source_queue_name, dest_bucket_name):
    """
    Pulls messages off a queue of calm VHS records, extracts the description 
    field from each record, and runs a Named Entity Recognition & Disambiguation 
    process on the text. The resulting list of subjects and entities is then 
    written to a specified s3 bucket alongside the source text.
    """
    s3 = boto3.client("s3")
    sqs = boto3.client("sqs")

    queue_url = sqs.get_queue_url(QueueName=source_queue_name)["QueueUrl"]

    queue_length = int(sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=["ApproximateNumberOfMessages"]
    )["Attributes"]["ApproximateNumberOfMessages"])

    loop = tqdm(get_texts_from_queue(queue_url), total=queue_length)

    for record_id, title, description in loop:
        key = "inferred_data/" + record_id + ".json"
        data_to_post = {
            "title": {
                "text": title,
                "entities": extract_entities(title)
            },
            "description": {
                "text": description,
                "entities": extract_entities(description)
            }
        }
        binary_data = json.dumps(data_to_post).encode("utf-8")
        s3.put_object(
            Body=binary_data,
            Bucket=dest_bucket_name,
            Key=key
        )


if __name__ == "__main__":
    process_queue()
