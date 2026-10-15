from __future__ import annotations

import random
from typing import Any, Dict, List

import numpy as np


def sample_random_hyperparameters(rng: random.Random | np.random.Generator | None = None) -> Dict[str, float]:
    if rng is None:
        rng = random.Random()
    if isinstance(rng, random.Random):
        return {
            "learning_rate": 10 ** rng.uniform(-5, -2),
            "weight_decay": 10 ** rng.uniform(-6, -2),
            "dropout": rng.uniform(0.0, 0.5),
            "batch_size": rng.choice([32, 64, 128]),
            "data_augmentation_strength": rng.uniform(0.0, 1.0),
        }
    return {
        "learning_rate": 10 ** float(rng.uniform(-5, -2)),
        "weight_decay": 10 ** float(rng.uniform(-6, -2)),
        "dropout": float(rng.uniform(0.0, 0.5)),
        "batch_size": int(rng.choice(np.array([32, 64, 128]))),
        "data_augmentation_strength": float(rng.uniform(0.0, 1.0)),
    }


def random_search(trials: int = 5, seed: int = 42) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    results = []
    for _ in range(trials):
        params = sample_random_hyperparameters(rng)
        results.append({"trial": len(results), **params})
    return results
