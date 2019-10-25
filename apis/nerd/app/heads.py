import torch
from torch import nn


class Labeller(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_dim = round(2 * (input_dim / 2) ** (1/2))

        self.fwd = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, input_embedding):
        return self.fwd(input_embedding).permute(1, 0, 2)


class Disambiguator(nn.Module):
    def __init__(self, input_dim, output_dim=200):
        super().__init__()
        layer_mult = (input_dim / output_dim) ** (1/3)
        hidden_dim_1 = round(output_dim * layer_mult)
        hidden_dim_2 = round(output_dim * (layer_mult ** 2))
        
        self.fwd = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim_2, output_dim)
        )

    def forward(self, input_embedding):
        return self.fwd(input_embedding).permute(1, 0, 2)