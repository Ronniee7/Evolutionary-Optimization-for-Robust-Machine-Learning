from __future__ import annotations

import argparse

from src.genetic_optimizer import run_genetic_search
from src.utils import load_config


def objective(chromosome):
    return 0.5 * chromosome.get("learning_rate", 1e-3) + 0.5 * chromosome.get("dropout", 0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a toy genetic hyperparameter search.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    bounds = {
        "learning_rate": (1e-5, 1e-2),
        "weight_decay": (1e-6, 1e-2),
        "dropout": (0.0, 0.5),
        "batch_size": (32, 128),
        "data_augmentation_strength": (0.0, 1.0),
    }
    result = run_genetic_search(
        bounds,
        objective,
        population_size=args.population or int(config.get("search", {}).get("population_size", 8)),
        generations=args.generations or int(config.get("search", {}).get("generations", 4)),
        mutation_rate=float(config.get("search", {}).get("mutation_rate", 0.2)),
        crossover_rate=float(config.get("search", {}).get("crossover_rate", 0.8)),
        seed=int(config.get("training", {}).get("seed", 42)),
    )
    print(result)


if __name__ == "__main__":
    main()
