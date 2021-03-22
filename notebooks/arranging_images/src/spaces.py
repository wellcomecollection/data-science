import lap
import numpy as np
from halo import Halo
from scipy.spatial.distance import cdist
from umap import UMAP


def make_grid(side_length):
    spacing = np.linspace(0, 1, side_length)
    grid = np.dstack(np.meshgrid(spacing, spacing)).reshape(-1, 2)
    return grid


def squash_embeddings(embeddings):
    spinner = Halo('squashing embeddings down to 2d and normalising').start()
    embeddings = UMAP(n_components=2).fit_transform(embeddings)
    normalised_embeddings = (
        (embeddings - embeddings.min(axis=0)) /
        (embeddings.max(axis=0) - embeddings.min(axis=0))
    )
    spinner.succeed()
    return normalised_embeddings


def get_assignments(embeddings_2d, side_length):
    spinner = Halo('finding best assignment of images to grid').start()
    grid = make_grid(side_length)
    cost = cdist(grid, embeddings_2d, 'sqeuclidean')
    _, row_assigns, _ = lap.lapjv(np.copy(cost))
    assignments = np.array(row_assigns).reshape(side_length, side_length)
    spinner.succeed()
    return assignments
