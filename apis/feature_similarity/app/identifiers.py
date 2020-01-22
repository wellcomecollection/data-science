import pickle

from .aws import get_object_from_s3

catalogue_ids = pickle.load(get_object_from_s3(
    'feature-similarity/2020-01-22/catalogue_ids.pkl'
))

catalogue_id_to_miro_id = pickle.load(get_object_from_s3(
    'feature-similarity/2020-01-22/catalogue_id_to_miro_id.pkl'
))


def miro_id_to_miro_uri(miro_id):
    return (
        "https://iiif.wellcomecollection.org/"
        f"image/{miro_id}.jpg/full/960,/0/default.jpg"
    )


def catalogue_id_to_catalogue_uri(catalogue_id):
    return 'https://wellcomecollection.org/works/' + catalogue_id


def expand_identifiers(catalogue_id):
    miro_id = catalogue_id_to_miro_id[catalogue_id]
    return {
        'catalogue_id': catalogue_id,
        'catalogue_uri': catalogue_id_to_catalogue_uri(catalogue_id),
        'miro_id': miro_id,
        'miro_uri': miro_id_to_miro_uri(miro_id)
    }


catalogue_id_set = set(catalogue_ids)

index_lookup = {id: ix for ix, id in enumerate(catalogue_ids)}
