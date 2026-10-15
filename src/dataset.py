from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from .utils import ensure_dir, set_seed


def get_default_transform(train: bool = False):
    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose([transforms.ToTensor(), normalize])


def get_dataset(name: str = "cifar10", root: str = "./data", train: bool = True, download: bool = True):
    if name.lower() == "cifar10":
        return datasets.CIFAR10(root=root, train=train, download=download, transform=get_default_transform(train=train))
    if name.lower() == "fashion-mnist" or name.lower() == "fashionmnist":
        return datasets.FashionMNIST(root=root, train=train, download=download, transform=get_default_transform(train=train))
    raise ValueError(f"Unsupported dataset '{name}'")


def build_dataloaders(config: dict, data_dir: str = "./data") -> Dict[str, DataLoader]:
    """Build train, validation, and test loaders with a deterministic validation split."""
    dataset_name = config.get("dataset", {}).get("name", "cifar10")
    root = config.get("dataset", {}).get("root", data_dir)
    batch_size = int(config.get("training", {}).get("batch_size", 64))
    validation_fraction = float(config.get("dataset", {}).get("validation_fraction", 0.1))
    seed = int(config.get("training", {}).get("seed", 42))
    num_workers = int(config.get("dataset", {}).get("num_workers", 2))

    ensure_dir(root)
    train_dataset = get_dataset(dataset_name, root=root, train=True, download=config.get("dataset", {}).get("download", True))
    test_dataset = get_dataset(dataset_name, root=root, train=False, download=config.get("dataset", {}).get("download", True))

    if validation_fraction <= 0:
        val_dataset = test_dataset
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        total_train = len(train_dataset)
        val_size = int(total_train * validation_fraction)
        train_size = total_train - val_size
        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return {"train": train_loader, "val": val_loader, "test": test_loader}
