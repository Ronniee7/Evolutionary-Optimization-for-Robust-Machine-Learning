from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import DataLoader

from .corruptions import apply_corruption
from .metrics import accuracy


def evaluate_clean(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)
    return float(correct / max(1, total))


def evaluate_corrupted(model: torch.nn.Module, loader: DataLoader, corruption_types: Iterable[str], severities: Iterable[int], device: torch.device) -> Dict[str, float]:
    model.eval()
    results = {}
    with torch.no_grad():
        for corruption_type in corruption_types:
            for severity in severities:
                correct = 0
                total = 0
                for inputs, targets in loader:
                    corrupted = torch.stack([apply_corruption(image, corruption_type, severity=severity) for image in inputs])
                    corrupted = corrupted.to(device)
                    logits = model(corrupted)
                    correct += (logits.argmax(1) == targets.to(device)).sum().item()
                    total += targets.size(0)
                results[f"{corruption_type}_{severity}"] = float(correct / max(1, total))
    return results


def evaluate_model_report(model: torch.nn.Module, loader: DataLoader, device: torch.device, corruption_types: Iterable[str], severities: Iterable[int]) -> Dict[str, Any]:
    clean = evaluate_clean(model, loader, device)
    corrupted = evaluate_corrupted(model, loader, corruption_types, severities, device)
    mean_corruption = float(sum(corrupted.values()) / max(1, len(corrupted)))
    return {"clean_accuracy": clean, "corruption_accuracy": mean_corruption, "per_corruption": corrupted}
