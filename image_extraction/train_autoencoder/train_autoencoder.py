from pathlib import Path

import typer

from src.dataset import AnomalyDetectionDataset
from src.model import AnomalyDetector


def main(
    data_dir: Path,
    hidden_dim: int = 64,
    n_hidden: int = 7,
    output_dim: int = 1024,
    n_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    model_file_path: Path = None,
    train_backbone: bool = False
):
    dataset = AnomalyDetectionDataset(data_dir)
    model = AnomalyDetector(
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        output_dim=output_dim,
        model_file_path=model_file_path
    )

    for parameter in model.backbone.parameters():
        parameter.requires_grad = train_backbone

    model.run_training_loop(
        dataset=dataset,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate
    )


if __name__ == "__main__":
    typer.run(main)
