from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def clean_corrupted_gap(clean_accuracy: float, corruption_accuracy: float) -> float:
    return float(clean_accuracy - corruption_accuracy)


def summarize_corruption_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    values = [float(item["accuracy"]) for item in records]
    if not values:
        return {"mean": 0.0, "std": 0.0}
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0}


def average_metric(metric_values: Iterable[float]) -> float:
    values = list(metric_values)
    return float(np.mean(values)) if values else 0.0


def evaluate_predictions(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    return {"accuracy": accuracy(logits, targets)}
