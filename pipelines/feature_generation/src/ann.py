from os.path import join

import nmslib
from halo import Halo

from .utils import get_data_dir


def build_ann_index(feature_vectors):
    print('\nBuilding nmslib index')
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(feature_vectors)
    index.createIndex({'post': 2}, print_progress=True)
    return index


def save_ann_index(ann_index):
    spinner = Halo(f'Saving nmslib index').start()
    data_dir = get_data_dir()
    ann_index.saveIndex(
        join(data_dir, 'feature_vectors.hnsw'), save_data=True
    )
    spinner.succeed()
