"""
Pulls messages off a queue of calm VHS records, extracts the description 
field from each record, and runs a Named Entity Recognition & Disambiguation 
process on the text. The resulting list of subjects and entities is then 
written to an s3 bucket alongside the source text.
"""
import os
import json
import boto3
from nerd.poll_queue import get_texts_from_queue
from nerd.extract import extract_entities


def process_queue():
    queue_url = os.environ['QUEUE_URL']
    bucket_name = "wellcomecollection-inference-calm"

    s3 = boto3.client("s3")

    for record_id, title, description in get_texts_from_queue(queue_url):
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
            Bucket=bucket_name,
            Key=key
        )


if __name__ == "__main__":
    process_queue()
