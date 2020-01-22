import json
import pickle
from os import mkdir
from os.path import exists, join, realpath, split

from halo import Halo
from tqdm import tqdm


def get_data_dir():
    file_dir, _ = split(realpath(__file__))
    data_dir = join(file_dir, 'data')
    if not exists(data_dir):
        mkdir(data_dir)
    return data_dir


def save_catalogue_ids(catalogue_ids):
    spinner = Halo(f'Saving catalogue ids').start()
    data_dir = get_data_dir()
    with open(join(data_dir, 'catalogue_ids.pkl'), 'wb') as f:
        pickle.dump(catalogue_ids, f)
    spinner.succeed()

def save_catalogue_miro_lookup(catalogue_id_to_miro_id):
    spinner = Halo(f'Saving catalogue_id to miro_id lookup').start()
    data_dir = get_data_dir()
    with open(join(data_dir, 'catalogue_id_to_miro_id.pkl'), 'wb') as f:
        pickle.dump(catalogue_id_to_miro_id, f)
    spinner.succeed()


def get_miro_to_catalogue_id_lookup(works_json_path):
    spinner = Halo(f'\nCounting lines in {works_json_path}').start()
    n_lines = sum(1 for line in open(works_json_path))
    spinner.succeed()

    miro_id_to_catalogue_id = {}
    print('\nLoading miro_id to catalogue_id lookups')
    with open(works_json_path) as f:
        for line in tqdm(f, total=n_lines):
            record = json.loads(line)
            if 'thumbnail' in record:
                url = record['thumbnail']['url']
                miro_id = url.split('/')[-5].split('.')[0]
                miro_id_to_catalogue_id[miro_id] = record['id']

    return miro_id_to_catalogue_id
