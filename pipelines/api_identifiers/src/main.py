from .aws import (get_assume_role_credentials, get_dynamo_client,
                  get_object_from_dynamo, get_object_from_s3, get_s3_client)
from .elastic import get_es_client
from .identifiers import (get_catalogue_id_miro, get_catalogue_id_sierra,
                          get_feature_index, get_palette_index)


def get_identifiers(miro_id):
    # get aws objects
    platform_dev_role_arn = 'arn:aws:iam::760097843905:role/platform-developer'
    platform_credentials = get_assume_role_credentials(platform_dev_role_arn)
    platform_dynamo = get_dynamo_client(platform_credentials)
    platform_s3 = get_s3_client(platform_credentials)

    # fetch metadata and identifiers
    is_cleared_for_catalogue_api = get_object_from_dynamo(
        platform_dynamo, 'vhs-sourcedata-miro', miro_id
    )['isClearedForCatalogueAPI']['BOOL']

    catalogue_id_sierra = get_catalogue_id_sierra(miro_id)
    catalogue_id_miro = get_catalogue_id_miro(
        platform_dynamo, platform_s3, miro_id
    )

    palette_index = get_palette_index(miro_id)
    feature_index = get_feature_index(miro_id)

    # combine data to build record for es
    identifiers = {
        'miro_id': miro_id,
        'is_cleared_for_catalogue_api': is_cleared_for_catalogue_api,
        'catalogue_id_sierra': catalogue_id_sierra,
        'catalogue_id_miro': catalogue_id_miro,
        'palette_index': palette_index,
        'feature_index': feature_index
    }
    return identifiers


def main(miro_id):
    identifiers = get_identifiers(miro_id)
    es_client = get_es_client()
    es_client.index(
        index='miro_identifiers',
        doc_type='_doc',
        body=identifiers
    )


if __name__ == "__main__":
    main(miro_id)
