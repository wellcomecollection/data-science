import math
from os.path import exists, expanduser, join
from pprint import pprint
from urllib.parse import quote, unquote

import more_itertools
import numpy as np
import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize
from scipy.spatial.distance import cosine

from .aws import get_wikidata_embedding


def tokenize(sentence):
    '''moses tokeniser'''
    seq = ' '.join(word_tokenize(sentence))
    seq = seq.replace(" n't ", "n 't ")
    return seq.split()


def split_list_into_chunks(input_list, chunk_size=50):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i+chunk_size]


def map_query_to_true_title(titles, normalized, redirects):
    query_title_to_true_title = {}
    for title in titles:
        true_title = title
        if true_title in normalized:
            true_title = normalized[true_title]
        if true_title in redirects:
            true_title = redirects[true_title]
        query_title_to_true_title[title] = true_title
    return query_title_to_true_title


def fetch_data_from_wikimedia_api(titles):
    query_url = (
        'https://en.wikipedia.org/w/api.php?action=query'
        '&format=json&redirects&prop=pageprops&titles='
    )
    normalized, redirects, pages = {}, {}, []
    for chunk in split_list_into_chunks(titles):
        url = query_url + '|'.join(chunk)
        response = requests.get(url).json()

        if 'normalized' in response['query']:
            normalized.update({
                title['from']: title['to']
                for title in response['query']['normalized']
            })
        if 'redirects' in response['query']:
            redirects.update({
                title['from']: title['to']
                for title in response['query']['redirects']
            })
        if 'pages' in response['query']:
            pages.extend(response['query']['pages'].values())

    return normalized, redirects, pages


def get_wikidata_ids(titles):
    normalized, redirects, pages = fetch_data_from_wikimedia_api(titles)
    query_to_true_title = map_query_to_true_title(
        titles, normalized, redirects
    )

    true_title_to_wikidata_id = {}
    for page in pages:
        title = page['title']
        try:
            wikidata_id = page['pageprops']['wikibase_item']
            true_title_to_wikidata_id[title] = wikidata_id
        except KeyError:
            pass

    query_to_wikidata_id = {
        query: true_title_to_wikidata_id[true_title]
        if true_title in true_title_to_wikidata_id
        else None
        for query, true_title
        in query_to_true_title.items()
    }

    return query_to_wikidata_id


def get_candidate_embeddings(entity):
    query_url = (
        'https://en.wikipedia.org/w/api.php?'
        'action=query&list=search&format=json&srsearch='
    )
    response = requests.get(query_url + entity).json()
    if 'suggestion' in response['query']['searchinfo']:
        suggestion = response['query']['searchinfo']['suggestion']
        return get_candidate_embeddings(suggestion)

    candidates = [item['title'] for item in response['query']['search']]
    candidate_wikidata_ids = get_wikidata_ids(candidates)

    embeddings = {}
    for title, wikidata_id in candidate_wikidata_ids.items():
        try:
            embeddings[title] = get_wikidata_embedding(wikidata_id)
        except ValueError:
            pass

    return embeddings


def calculate_candidate_relevance(embeddings, prediction, alpha):
    similarity = pd.Series({
        candidate: cosine(embedding, prediction) * math.exp(alpha * rank)
        for rank, (candidate, embedding) in enumerate(embeddings.items())
    })
    return similarity.sort_values()


def get_url_dict(token_seq, pred_embeddings, link_indexes, alpha):
    url_dict = {}
    for group in more_itertools.consecutive_groups(link_indexes):
        group = list(group)
        start, end = group[0], group[-1] + 1
        mean = (
            pred_embeddings[0, start:end]
            .mean(dim=0)
            .detach().cpu().numpy()
        )

        entity = ' '.join(token_seq[start:end])
        candidates = get_candidate_embeddings(entity)

        relevance = calculate_candidate_relevance(candidates, mean, alpha)
        # pprint(relevance)

        try:
            best_candidate = relevance.index.values[0]
            url = 'https://en.wikipedia.org/wiki/' + quote(best_candidate)
            url_dict[entity] = url
        except IndexError:
            pass

    return url_dict


def add_links_to_text(token_seq, url_dict):
    output_html = ' '.join(token_seq)
    entities = list(url_dict.keys())
    sorted_entities = sorted(entities, key=len, reverse=True)

    for entity in sorted_entities:
        hashed_entity = ' ' + str(hash(entity)) + ' '
        entity = ' ' + entity + ' '
        output_html = output_html.replace(entity, hashed_entity)

    for entity, url in url_dict.items():
        hashed_entity = ' ' + str(hash(entity)) + ' '
        html_element = (
            ' <a class="bg-white br2 ph1 code black no-underline f6 b" '
            f'target="_blank" href="{url}">{entity}</a> '
        )
        output_html = output_html.replace(hashed_entity, html_element)

    return output_html
