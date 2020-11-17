from pathlib import Path
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from weco_datascience.logging import get_logger

from .autoencoder import Autoencoder
from .feature_extractor import FeatureExtractor
from .training import choose_device, prepare_dataloaders, prepare_optimiser

log = get_logger(__name__)


class AnomalyDetector(nn.Module):
    def __init__(self, hidden_dim, n_hidden, output_dim, device=choose_device(), model_file_path=None):
        super().__init__()
        self.device = device

        log.info(f"Initialising AnomalyDetector on device: {self.device.type}")
        self.backbone = FeatureExtractor()
        self.head = Autoencoder(
            input_dim=self.backbone.output_dim,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden
        )
        if model_file_path:
            self.model_file_path = model_file_path
            self.load_state_dict(torch.load(model_file_path))
        else:
            self.model_file_path = Path("/data/models/AnomalyDetector.pt")

        if not self.model_file_path.parent.exists():
            Path.mkdir(self.model_file_path.parent, parents=True)

        self.to(self.device)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def get_loss_value(self, losses):
        return np.mean(losses[-20:])

    def _train(self):
        self.train()
        for i, inputs in enumerate(self.train_dataloader):
            inputs = inputs.to(self.device)

            self.optimiser.zero_grad()
            targets = self.backbone(inputs)
            preds = self.head(targets)

            loss = self.loss_function(preds, targets)
            loss.backward()
            self.optimiser.step()

            self.losses.append(loss.item())
            log.info(
                f"Batch {i+1}/{len(self.train_dataloader)}, "
                f"loss: {self.get_loss_value(self.losses)}"
            )

    def _validate(self):
        self.eval()
        with torch.no_grad():
            for i, inputs in enumerate(self.test_dataloader):
                inputs = inputs.to(self.device)

                targets = self.backbone(inputs)
                preds = self.head(targets)

                loss = self.loss_function(preds, targets)
                self.val_losses.append(loss.item())
                log.info(
                    f"Batch {i+1}/{len(self.test_dataloader)}, "
                    f"loss: {self.get_loss_value(self.losses)}"
                )

    def run_training_loop(self, dataset, batch_size, n_epochs, learning_rate, ):
        log.info("Preparing model for training")
        self.train_dataloader, self.test_dataloader = prepare_dataloaders(
            dataset, batch_size
        )
        self.optimiser = prepare_optimiser(self.parameters(), learning_rate)
        self.loss_function = nn.MSELoss()

        self.losses, self.val_losses = [], []
        log.info(f"Starting {n_epochs} training epochs")
        for epoch in range(n_epochs):
            log.info(f"Training, epoch {epoch+1}")
            self._train()
            log.info(f"Validating")
            self._validate()

        log.info(f"Saving model state_dict at {self.model_file_path}")
        torch.save(self.state_dict(), self.model_file_path)
