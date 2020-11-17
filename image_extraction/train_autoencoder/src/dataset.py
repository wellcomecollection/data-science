import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from weco_datascience.logging import get_logger

log = get_logger(__name__)


class AnomalyDetectionDataset(Dataset):
    def __init__(self, data_dir):
        log.info(f"Initialising Dataset from path {data_dir}")
        self.data_paths = [
            Path(root) / file_name
            for root, _, files in os.walk(data_dir)
            for file_name in files
            if file_name != "ledger.json"
        ]
        log.info(f"Found {len(self.data_paths)} data points")

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        image = (
            Image.open(data_path)
            .convert('RGB')
            .resize((224, 224), resample=Image.BILINEAR)
        )
        return ToTensor()(image)

    def __len__(self):
        return len(self.data_paths)
