from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gadir.config import load_experiment_config
from gadir.training.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA, LoRA-GA, or GADI-R experiments.")
    parser.add_argument("--config", required=True, help="Path to an experiment yaml file.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
