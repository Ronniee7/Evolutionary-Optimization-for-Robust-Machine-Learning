from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset import build_dataloaders
from src.model import SmallCNN
from src.train import fit_model
from src.utils import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the default baseline model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size

    set_seed(int(config.get("training", {}).get("seed", 42)))
    loaders = build_dataloaders(config)
    result = fit_model(config, loaders["train"], loaders["val"])
    print({"val_accuracy": result["val_accuracy"], "training_time": result["training_time"]})


if __name__ == "__main__":
    main()
