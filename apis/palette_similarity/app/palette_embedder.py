import numpy as np
import torch
from torch import nn

from .aws import download_object_from_s3
from .colours import hex_to_rgb, rgb_to_lab


class PaletteEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_transform = nn.Sequential(
            nn.Linear(3, 6), nn.ReLU(),
            nn.Linear(6, 12)
        )
        self.embedder = nn.Sequential(
            nn.Linear(60, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 30)
        )

    def forward(self, input_palettes):
        batch_size = input_palettes.shape[0]
        intermediate = self.initial_transform(input_palettes)
        flattened = intermediate.reshape(batch_size, -1)
        embedded = self.embedder(flattened)
        return embedded


# create palette embedding model
download_object_from_s3('palette/model_state_dict.pt')
palette_embedder = PaletteEmbedder().eval()
palette_embedder.load_state_dict(torch.load(
    'model_state_dict.pt',
    map_location=torch.device('cpu')
))


def embed_hex_palette(palette):
    '''
    takes a list of hex strings, reformats them as a 5x3 colour palette tensor, 
    and passes them through a pytorch palette embedding model. This produces a 
    1-D palette embedding which can be compared to a catalogue of pre-computed 
    palette embeddings
    '''
    rgb_palette = np.array([hex_to_rgb(colour) for colour in palette])
    lab_palette = rgb_to_lab(rgb_palette)
    palette_tensor = torch.Tensor(lab_palette).unsqueeze(0)
    query_embedding = palette_embedder(palette_tensor).detach().numpy()
    return query_embedding
