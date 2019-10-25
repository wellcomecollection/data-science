import io
import boto3
import torch
import pickle
from os import path
from bs4 import BeautifulSoup
from .model import LinkLabeller
from .disambiguate import get_identifiers

device = torch.device("cpu")

ix_to_token = pickle.load(open(
    'data/ix_to_token.pkl', 'rb'
))

unique_characters = pickle.load(open(
    'data/unique_characters.pkl', 'rb'
))

embedding_matrix = torch.load(
    'data/embedding_matrix.pt',
    map_location=device
)

model = LinkLabeller(
    unique_characters=unique_characters,
    word_vector_embedding_matrix=embedding_matrix
)

model.load_state_dict(torch.load(
    'data/model_state_dict.pt',
    map_location=device
))


def _extract_entities(document, remove_disambiguation_pages=True):
    data = {
        'title': 'title',
        'description': 'description',
        'wikipedia_url': 'url',
        'wikidata_id': 'wikidata_id',
        'lcsh_id': 'lcsh_id',
        'mesh_id': 'mesh_id'
    }
    return {'Francis Crick': data}


def extract_entities(document, remove_disambiguation_pages=True):
    text = BeautifulSoup(document, features="html.parser").text.strip()
    if text == '':
        entities = []
    else:
        try:
            entities = model.find_entities(text)
        except RuntimeError:
            entities = []

    with_identifiers = {}
    for entity in entities:
        entity_data = get_identifiers(entity)
        if entity_data is not None:
            with_identifiers[entity] = entity_data

    if remove_disambiguation_pages:
        with_identifiers = {
            entity: data
            for entity, data in with_identifiers.items()
            if 'disambiguation page' not in str(data['description'])
        }

    return with_identifiers
