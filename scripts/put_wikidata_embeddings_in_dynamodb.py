#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
We use biggraph wikidata embeddings (https://ai.facebook.com/blog/open-sourcing-pytorch-biggraph-for-faster-embeddings-of-extremely-large-graphs/) 
in our named entity recognition & disambiguation model. Each embedding is
unpacked from the original .tar.gz file and split out into an individual file
before this script puts them into the wikidata-biggraph-embeddings dynamodb table.

The embeddings are stored and retrieved as bytestrings (via numpy), making them
fast/easy to load back into pytorch. That loading process is outlined in the 
comment at the end of this script.
"""
import os
import boto3
import torch
from tqdm import tqdm

base_path = '/storage/data/wikidata_embeddings/'
wikidata_ids = os.listdir(base_path)

table = (
    boto3
    .resource('dynamodb', 'eu-west-1')
    .Table('wikidata-biggraph-embeddings')
)

with table.batch_writer() as batcher:
    for wikidata_id in tqdm(wikidata_ids):
        wikidata_embedding = torch.load(base_path + wikidata_id)
        embedding_as_bytes = wikidata_embedding.numpy().tobytes()
        batcher.put_item(
            Item={
                'wikidata_id': wikidata_id,
                'embedding': embedding_as_bytes
            }
        )


# to get embeddings out again

# dynamo = boto3.client('dynamodb', 'eu-west-1')
#
# def get_wikidata_embedding(wikidata_id):
#     response = dynamo.get_item(
#         TableName='wikidata-biggraph-embeddings',
#         Key={'wikidata_id': {'S': wikidata_id}}
#     )
#
#     array_as_bytes = response['Item']['embedding']['B']
#
#     unpacked_embedding = torch.Tensor(np.frombuffer(
#         array_as_bytes, dtype=np.float32
#     ))
#     return unpacked_embedding
