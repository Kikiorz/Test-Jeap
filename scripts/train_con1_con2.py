#!/usr/bin/env python3
"""Launch one immutable phase of the Con1/Con2 training sequence.

This wrapper removes an easy-to-miss failure mode in which the Con1 full phase
silently reloads the released JEPA-WAM checkpoint instead of the completed
5k Change warm-up.  Each phase therefore requires an explicit source params
directory and constructs the appropriate strict loader in code.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import train as train_lib

from openpi.training import config as config_lib
from openpi.training import weight_loaders


CONFIGS = {
    "con1-warmup": "pi05_libero_vjepa_con1_warmup",
    "con1-full": "pi05_libero_vjepa_con1",
    "con2-offline": "pi05_libero_vjepa_con2_adapter",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=tuple(CONFIGS))
    parser.add_argument("--source-params", type=Path, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--checkpoint-base-dir", type=Path, required=True)
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=None,
        help="Optional final global step override (useful for bounded continuation runs).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional global batch-size override for a bounded continuation run.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_params.exists():
        raise FileNotFoundError(f"Source params do not exist: {args.source_params}")
    config = config_lib.get_config(CONFIGS[args.phase])
    if args.phase in ("con1-warmup", "con1-full"):
        loader = weight_loaders.ActionChangeCheckpointWeightLoader(str(args.source_params))
    else:
        loader = weight_loaders.CheckpointWeightLoader(
            str(args.source_params),
            missing_regex=".*change_to_action_(k|v)_(down|up).*",
        )
    config = dataclasses.replace(
        config,
        exp_name=args.exp_name,
        checkpoint_base_dir=str(args.checkpoint_base_dir),
        weight_loader=loader,
        resume=args.resume,
        overwrite=args.overwrite,
        wandb_enabled=args.wandb,
        **({"num_train_steps": args.num_train_steps} if args.num_train_steps is not None else {}),
        **({"batch_size": args.batch_size} if args.batch_size is not None else {}),
    )
    train_lib.main(config)


if __name__ == "__main__":
    main()
