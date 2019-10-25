import json
from os.path import dirname, expanduser, realpath

import torch
from torch import nn, optim
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)

from .character_level_network import CharacterLevelNetwork
from .aws import get_object_from_s3


config = json.loads(get_object_from_s3('config.json').decode('utf-8'))


class Backbone(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # first dim of self.wv_embedding is len(vocabulary)
        wv_embedding_dim = 300
        self.wv_embedding = nn.Embedding(
            num_embeddings=config['n_words'],
            embedding_dim=wv_embedding_dim
        )

        # input dim of self.cln is len(unique_characters)
        self.cln = CharacterLevelNetwork(input_dim=config['n_chars'])

        word_level_lstm_input_dim = (
            wv_embedding_dim +
            (self.cln.output_dim * 2)
        )

        self.word_level_lstm = nn.LSTM(
            input_size=word_level_lstm_input_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            num_layers=2,
            dropout=0.2
        )

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

    def forward(self, c_seqs, t_seqs):
        c_lens, t_lens = self.get_seq_lens(c_seqs, t_seqs)
        exit_seqs = self.get_exit_seqs(c_seqs)
        wv_seqs = self.wv_embedding(t_seqs)

        char_fwd, char_bwd = self.cln(c_seqs, exit_seqs, c_lens)

        concats = torch.cat([char_fwd, char_bwd, wv_seqs], dim=2)
        sorted_lens, sort_ixs = t_lens.sort(dim=0, descending=True)
        concats = torch.stack([concats[i] for i in sort_ixs])

        packed = pack_padded_sequence(
            concats,
            lengths=sorted_lens,
            batch_first=True
        )

        packed_embedded, _ = self.word_level_lstm(packed)
        embedded, _ = pad_packed_sequence(packed_embedded)
        embedded = self.head(embedded)
        return embedded, sort_ixs

    def get_exit_seqs(self, c_seqs):
        # indexes of token separators (ie char_to_ix[' '])
        exit_seqs = [
            (seq == config['exit_ix']).nonzero().squeeze() for seq in c_seqs
        ]
        return pad_sequence(sequences=exit_seqs, batch_first=True)

    def get_seq_lens(self, c_seqs, t_seqs):
        # seq length of non-padding chars (ie char_to_ix['xxpad'])
        c_lens = torch.LongTensor([
            len([char for char in seq if char != config['char_pad']])
            for seq in c_seqs
        ])
        # seq length of non-padding tokens (ie token_to_ix['xxpad'])
        t_lens = torch.LongTensor([
            len([token for token in seq if token != config['token_pad']])
            for seq in t_seqs
        ])

        return c_lens, t_lens
