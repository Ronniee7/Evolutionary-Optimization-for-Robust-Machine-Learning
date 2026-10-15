from __future__ import annotations

import argparse

from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Final evaluation entry point for the project.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Final evaluation configured for dataset: {config.get('dataset', {}).get('name', 'cifar10')}")


if __name__ == "__main__":
    main()
