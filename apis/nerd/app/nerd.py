import json
import pickle
from os.path import expanduser
from pprint import pprint

import numpy as np
import torch
from torch import nn

from .backbone import Backbone
from .aws import get_object_from_s3
from .heads import Disambiguator, Labeller
from .model_utils import (add_links_to_text, get_url_dict, get_wikidata_ids,
                          tokenize)

token_to_ix = pickle.loads(get_object_from_s3('token_to_ix.pkl'))
char_to_ix = pickle.loads(get_object_from_s3('char_to_ix.pkl'))


class NERD(nn.Module):
    def __init__(self, backbone_dim=2048, alpha=0.8):
        super().__init__()
        self.backbone = Backbone(backbone_dim)
        self.labeller = Labeller(backbone_dim // 2)
        self.disambiguator = Disambiguator(backbone_dim // 2)
        self.alpha = alpha

    def forward(self, c_seqs, t_seqs):
        backboned, sort_indicies = self.backbone(c_seqs, t_seqs)

        labels = self.labeller(backboned).permute(0, 2, 1)
        embeddings = self.disambiguator(backboned)

        return labels, embeddings, sort_indicies

    def get_model_inputs(self, token_seq):
        c_seq = torch.LongTensor(np.array(
            [char_to_ix['xxbos'], char_to_ix[' ']] +
            [char_to_ix[char]
             if char in char_to_ix
             else char_to_ix['xxunk']
             for char in ' '.join(token_seq)] +
            [char_to_ix[' '], char_to_ix['xxeos']]
        )).unsqueeze(0)

        t_seq = torch.LongTensor(np.array([
            token_to_ix[token]
            if token in token_to_ix
            else token_to_ix['xxunk']
            for token in token_seq
        ])).unsqueeze(0)

        return c_seq, t_seq

    def annotate(self, text, alpha=None):
        alpha = alpha or self.alpha
        tokens = tokenize(text)
        c_seq, t_seq = self.get_model_inputs(tokens)

        pred_labels, pred_embeddings, _ = self.forward(c_seq, t_seq)

        labels = pred_labels.squeeze().detach().numpy().argmax(0)
        link_indexes = np.where(labels == 1)[0].tolist()
        url_dict = get_url_dict(tokens, pred_embeddings, link_indexes, alpha)
        output_html = add_links_to_text(tokens, url_dict)
        return output_html

    def extract_entities(self, text, alpha=None):
        alpha = alpha or self.alpha
        tokens = tokenize(text)
        c_seq, t_seq = self.get_model_inputs(tokens)

        pred_labels, pred_embeddings, _ = self.forward(c_seq, t_seq)
        labels = pred_labels.squeeze().detach().numpy().argmax(0)
        link_indexes = np.where(labels == 1)[0].tolist()

        url_dict = get_url_dict(tokens, pred_embeddings, link_indexes, alpha)
        titles = [
            url.replace('https://en.wikipedia.org/wiki/', '')
            for url in url_dict.values()
        ]
        wikidata_ids = {
            name: wikidata_id
            for name, wikidata_id in get_wikidata_ids(titles).items()
            if wikidata_id is not None
        }
        return json.dumps(wikidata_ids)
