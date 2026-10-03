from __future__ import annotations

import argparse
import sys

from .unet import cli as unet_cli
from .dlinknet import cli as dlinknet_cli
from .csnet import cli as csnet_cli
from .bisenetv2 import cli as bisenetv2_cli
from .ddrnet import cli as ddrnet_cli
from .pidnet import cli as pidnet_cli
from .segformer import cli as segformer_cli

MODEL_MODULES = {
    "unet": unet_cli,
    "dlinknet": dlinknet_cli,
    "csnet": csnet_cli,
    "bisenetv2": bisenetv2_cli,
    "ddrnet": ddrnet_cli,
    "pidnet": pidnet_cli,
    "segformer": segformer_cli,
}


def main():
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="Unified Benchmark Suite for SOAR Comparative Baselines (Q1 Standard)",
    )
    parser.add_argument(
        "model",
        choices=list(MODEL_MODULES.keys()),
        help=f"Target benchmark architecture: {list(MODEL_MODULES.keys())}",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Subcommands and arguments passed to the specific model runner (e.g. train, val, predict)",
    )

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        parser.parse_args(sys.argv[1:])
        return

    model_arg = sys.argv[1].lower()
    if model_arg in MODEL_MODULES:
        # Pass remaining arguments to model-specific CLI
        sys.argv = [f"benchmark {model_arg}"] + sys.argv[2:]
        MODEL_MODULES[model_arg].main()
    else:
        parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    main()
