from __future__ import annotations

import time
from typing import Any, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import SmallCNN


def get_device(device: str | None = None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = str(config.get("optimizer", "adam")).lower()
    lr = float(config.get("learning_rate", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_seen += batch_size
        total_correct += (logits.argmax(1) == targets).sum().item()
        total_loss += loss.item() * batch_size

    loss = total_loss / max(1, total_seen)
    accuracy = total_correct / max(1, total_seen)
    return {"loss": loss, "accuracy": accuracy}


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            total_correct += (logits.argmax(1) == targets).sum().item()
            total_seen += inputs.size(0)
    return {"accuracy": total_correct / max(1, total_seen)}


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: Dict[str, Any], device: str | None = None) -> Dict[str, Any]:
    device = get_device(device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config.get("training", {}))
    history = []
    start_time = time.time()

    for epoch in range(int(config.get("training", {}).get("epochs", 1))):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"], "val_accuracy": val_metrics["accuracy"]})

    elapsed = time.time() - start_time
    final = evaluate_model(model, val_loader, device)
    return {"model": model, "history": history, "val_accuracy": final["accuracy"], "training_time": elapsed}


def fit_model(config: Dict[str, Any], train_loader: DataLoader, val_loader: DataLoader, device: str | None = None) -> Dict[str, Any]:
    model = SmallCNN.from_config(config)
    return train_model(model, train_loader, val_loader, config, device=device)
