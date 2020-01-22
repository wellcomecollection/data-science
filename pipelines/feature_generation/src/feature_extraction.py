from io import BytesIO

import torch
from halo import Halo
from PIL import Image
from torchvision import transforms
from torchvision.models.vgg import vgg16

from .aws import get_object_from_s3, put_object_to_s3

device = (
    torch.device('cuda')
    if torch.cuda.is_available()
    else torch.device('cpu')
)

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

feature_extractor = vgg16(pretrained=True).to(device).eval()
feature_extractor.classifier = feature_extractor.classifier[:4]


def extract_features(image):
    image_tensor = transform_pipeline(image)
    features = feature_extractor(image_tensor.unsqueeze(0))
    return features.squeeze().detach().cpu().numpy()


def extract_and_save_image_features(miro_id, object_key):
    spinner = Halo(f'Feature extraction - {miro_id}').start()
    try:
        image = Image.open(BytesIO(get_object_from_s3(
            object_key=object_key,
            bucket_name='wellcomecollection-miro-images-public',
            profile_name='platform-dev'
        )))
        feature_vector = extract_features(image)
        put_object_to_s3(
            binary_object=feature_vector.tobytes(),
            key='feature_vectors/' + miro_id,
            bucket_name='miro-images-feature-vectors',
            profile_name='data-dev'
        )
        spinner.succeed()
    except:
        spinner.fail(f'Feature extraction failed - {miro_id}')
        pass
