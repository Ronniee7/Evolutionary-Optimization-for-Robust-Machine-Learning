from __future__ import annotations

import argparse

from src.random_search import random_search
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random-search candidates.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    trials = args.trials or int(config.get("search", {}).get("random_trials", 5))
    results = random_search(trials=trials, seed=int(config.get("training", {}).get("seed", 42)))
    print(results)


if __name__ == "__main__":
    main()
