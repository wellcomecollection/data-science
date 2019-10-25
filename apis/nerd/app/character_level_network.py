import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CharacterLevelNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.embedding_dim = output_dim // 2

        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=self.embedding_dim
        )

        self.char_level_lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=output_dim,
            bidirectional=True,
        )

        self.head_fwd = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim, output_dim),
        )

        self.head_bwd = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, char_seqs, exit_seqs, lengths):
        x = self.embedding(char_seqs)

        x = pack_padded_sequence(x, lengths=lengths, batch_first=True)

        x, _ = self.char_level_lstm(x)
        out, _ = pad_packed_sequence(x, batch_first=True)

        # pop out the character embeddings at position of the end of each token
        out = torch.stack([out[i, exit_seqs[i]] for i in range(len(out))])

        out_fwd, out_bwd = torch.chunk(out, 2, 2)

        pred_fwd = self.head_fwd(out_fwd[:, 1:])
        pred_bwd = self.head_bwd(out_bwd[:, :-1])

        return pred_fwd, pred_bwd
