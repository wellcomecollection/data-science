import os
import pickle
from urllib.parse import unquote_plus

import numpy as np
import torch
from nltk.tokenize import word_tokenize

from .sentence_encoder import SentenceEncoder
from .aws import get_object_from_s3, download_object_from_s3

# Load model data (don't fetch from s3 if developing locally)
if 'DEVELOPMENT' in os.environ:
    base_path = os.path.expanduser('~/datasets/devise_search/')
    model = SentenceEncoder()
    model.load_state_dict(torch.load(
        os.path.join(base_path, 'sentence-encoder-2018-10-16.pt'),
        map_location='cpu'
    ))
    word_to_index = pickle.load(open(
        os.path.join(base_path, 'word_to_index.pkl'),
        'rb'
    ))
    index_to_wordvec = np.load(
        os.path.join(base_path, 'index_to_wordvec.npy')
    )
else:
    model = SentenceEncoder()
    model.load_state_dict(torch.load(
        get_object_from_s3('devise_search_api/sentence-encoder-2018-10-16.pt'),
        map_location='cpu'
    ))
    word_to_index = pickle.load(
        get_object_from_s3('devise_search_api/word_to_index.pkl'),
        'rb'
    )
    index_to_wordvec = np.load(
        get_object_from_s3('devise_search_api/index_to_wordvec.npy')
    )


def embed(query_text, model=model, word_to_index=word_to_index, index_to_wordvec=index_to_wordvec):
    clean_text = unquote_plus(query_text)
    word_tokens = ['<s>'] + word_tokenize(clean_text.lower()) + ['</s>']
    word_vector_tensor = torch.Tensor(np.stack([[
        index_to_wordvec[word_to_index[word]]
        for word in word_tokens
        if word in word_to_index
    ]]))

    embedding = model(word_vector_tensor)

    return embedding.detach().numpy().squeeze()


def id_to_url(image_id):
    return f'https://iiif.wellcomecollection.org/image/{image_id}.jpg/full/960,/0/default.jpg'
