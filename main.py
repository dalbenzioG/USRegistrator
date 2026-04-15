from __future__ import annotations

import argparse

from train import run_training


DEFAULT_CONFIG = "configs/deepreg_synth.yaml"


def main():
    parser = argparse.ArgumentParser(
        description="Run the USRegistrator training pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=(
            "Optional config override. The recommended workflow is to edit "
            f"{DEFAULT_CONFIG} and run `python main.py`."
        ),
    )
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
