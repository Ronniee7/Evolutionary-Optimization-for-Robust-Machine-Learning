"""Evolutionary optimization for robust machine learning."""

__all__ = [
    "SmallCNN",
    "set_seed",
    "load_config",
    "ensure_dir",
    "build_dataloaders",
    "apply_corruption",
    "evaluate_model",
    "train_model",
    "evaluate_fitness",
    "GeneticOptimizer",
]

__version__ = "0.1.0"

from .model import SmallCNN
from .utils import ensure_dir, load_config, set_seed
