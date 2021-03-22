
from io import BytesIO

import numpy as np
import torch
from halo import Halo
from PIL import Image
from torchvision import transforms
from torchvision.models.vgg import vgg16
from tqdm import tqdm

from .file_utils import file_names_in_dir

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


def extract_feature_vector(image):
    image_tensor = transform_pipeline(image)
    features = feature_extractor(image_tensor.unsqueeze(0))
    return features.squeeze().detach().cpu().numpy()


def find_unprocessed_images(image_dir, feature_dir):
    spinner = Halo('finding unprocessed images').start()
    image_ids = file_names_in_dir(image_dir)
    feature_ids = file_names_in_dir(feature_dir)
    unprocessed_images = image_ids - feature_ids
    spinner.succeed()
    return unprocessed_images


def extract_image_features(image_dir, feature_dir):
    unprocessed_image_ids = find_unprocessed_images(image_dir, feature_dir)

    if len(unprocessed_image_ids) == 0:
        print('No new features to extract!')
    else:
        loop = tqdm(unprocessed_image_ids)
        for image_id in loop:
            loop.set_description(f'extracting features from image: {image_id}')
            image = Image.open(f'{image_dir}/{image_id}.jpg')
            image_features = extract_feature_vector(image)

            np.save(
                file=f'{feature_dir}/{image_id}.npy',
                arr=image_features
            )
