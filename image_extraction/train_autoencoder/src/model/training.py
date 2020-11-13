import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


def choose_device():
    use_gpu = torch.cuda.is_available()
    return torch.device("cuda" if use_gpu else "cpu")


def prepare_dataloaders(dataset, batch_size, train_test_split=0.8):
    train_length = int(train_test_split * len(dataset))
    test_length = len(dataset) - train_length
    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[train_length, test_length],
        # set seed to ensure consistent split across training runs
        generator=torch.Generator().manual_seed(42)
    )
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def prepare_optimiser(model_parameters, learning_rate):
    trainable_parameters = filter(lambda p: p.requires_grad, model_parameters)
    return Adam(trainable_parameters, lr=learning_rate)
