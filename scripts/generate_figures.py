from __future__ import annotations

import argparse

from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate analysis plots for the project.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Figures will be saved under: {config.get('logging', {}).get('plots_dir', './plots')}")


if __name__ == "__main__":
    main()
