import os
import pickle

import requests
from tqdm import tqdm

DATA_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'miro_id_to_catalogue_id.pkl'
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'catalogue_id_to_miro_id.pkl'
)

with open(DATA_PATH, 'rb') as f:
    identifiers = pickle.load(f)

if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, 'rb') as f:
        catalogue_id_to_miro_id = pickle.load(f)
else:
    catalogue_id_to_miro_id = {}

for i, (miro_id, miro_catalogue_id) in enumerate(tqdm(identifiers.items())):
    if miro_catalogue_id in catalogue_id_to_miro_id:
        pass
    else:
        base_url = 'https://api.wellcomecollection.org/catalogue/v2/works/'
        response = requests.get(base_url + str(miro_catalogue_id))
        sierra_catalogue_id = response.url.split('/')[-1]
        catalogue_id_to_miro_id[miro_catalogue_id] = miro_id
        catalogue_id_to_miro_id[sierra_catalogue_id] = miro_id
        if i % 500 == 0:
            with open(OUTPUT_PATH, 'wb') as f:
                pickle.dump(catalogue_id_to_miro_id, f)

with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(catalogue_id_to_miro_id, f)
