import numpy as np
from tqdm import tqdm
from weco_datascience.logging import get_logger
from torch import nn

from .autoencoder import Autoencoder
from .feature_extractor import FeatureExtractor
from .training import choose_device, prepare_dataloaders, prepare_optimiser

log = get_logger(__name__)


class AnomalyDetector(nn.Module):
    def __init__(self, hidden_dim, n_hidden, output_dim, device=choose_device()):
        super().__init__()
        self.device = device
        log.info(f"Initialising AnomalyDetector on device: {self.device.type}")
        self.backbone = FeatureExtractor()
        self.head = Autoencoder(
            input_dim=self.backbone.output_dim,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden
        )
        self.to(self.device)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def run_training(self, dataset, batch_size, n_epochs, learning_rate):
        log.info("Preparing dataloaders for training")
        train_dataloader, test_dataloader = prepare_dataloaders(
            dataset, batch_size
        )

        log.info("Preparing optimiser for training")
        optimiser = prepare_optimiser(self.parameters(), learning_rate)

        log.info("Preparing loss function for training")
        loss_function = nn.MSELoss()

        self.losses = []
        for epoch in range(n_epochs):
            self.train()
            loop = tqdm(train_dataloader)
            for inputs in loop:
                inputs = inputs.to(self.device)

                optimiser.zero_grad()
                targets = self.backbone(inputs)
                preds = self.head(targets)

                loss = loss_function(preds, targets)
                loss.backward()
                optimiser.step()

                self.losses.append(loss.item())
                loop.set_description('Epoch {}/{}'.format(epoch + 1, n_epochs))
                loop.set_postfix(loss=np.mean(self.losses[-20:]))
