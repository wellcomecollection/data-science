import logging
import os

import daiquiri
from botocore.exceptions import ClientError
from torchvision.models.vgg import vgg16

from src.aws import download_object_from_s3

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger("prestart")

try:
    logger.info('Fetching pretrained VGG16 model')
    feature_extractor = vgg16(pretrained=True, progress=False)
    logger.info('Fetched pretrained VGG model')
except Exception as e:
    logger.error(f'Failed to fetch pretrained VGG model: {e}')
    raise

try:
    logger.info('Fetching pretrained LSHEncoder model')
    download_object_from_s3(
        object_key=os.environ['MODEL_OBJECT_KEY'],
        bucket_name='model-core-data',
        profile_name='data-dev'
    )
    logger.info('Fetched pretrained LSHEncoder model')
except ClientError as e:
    logger.error(f'Failed to fetch pretrained LSHEncoder: {e}')
    raise
