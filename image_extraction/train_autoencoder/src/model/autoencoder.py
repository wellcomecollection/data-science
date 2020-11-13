from torch import nn
import numpy as np
from weco_datascience.logging import get_logger

log = get_logger(__name__)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_hidden, dropout=0.4):
        super().__init__()
        self.dropout = dropout
        self.layer_dims = self.calculate_layer_dims(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
        )
        self.layers = self.build_layers()
        log.info(f"Initialising {self}")

    def build_layers(self):
        layers = []
        for i in range(len(self.layer_dims)-1):
            in_dim, out_dim = self.layer_dims[i], self.layer_dims[i+1]
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout)
            ])

        layers = layers[:-2]  # don't want relu or dropout on the final layer
        return nn.Sequential(*layers)

    def calculate_layer_dims(self, input_dim, hidden_dim, n_hidden):
        """
        calculate an hourglass-shaped set of layers which satisfy the
        supplied dimension parameters
        """
        n_encoder = n_hidden // 2
        n_decoder = n_hidden - (n_hidden // 2)
        encoder_layers = np.linspace(
            start=input_dim,
            stop=hidden_dim,
            num=n_encoder,
            endpoint=False
        )
        decoder_layers = np.linspace(
            start=hidden_dim,
            stop=input_dim,
            num=n_decoder
        )
        layers = np.concatenate([encoder_layers, decoder_layers]).astype(int)
        return layers

    def __repr__(self):
        return f"Autoencoder({self.layer_dims})"

    def forward(self, x):
        return self.layers(x)
