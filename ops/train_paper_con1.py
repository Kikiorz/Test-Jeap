#!/usr/bin/env python3
"""Run a resumable segment without resetting its phase-specific LR schedule."""

import argparse
import dataclasses
import importlib.util
from pathlib import Path

from openpi.training.config import paper_con1_config, paper_con1_mean_from5k_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--until-step", type=int, required=True)
    parser.add_argument("--late2", action="store_true")
    parser.add_argument("--direct-delta", action="store_true")
    parser.add_argument("--mean-from5k", action="store_true",
                        help="Separate mean-feature loss phase from 5k weights; freeze front sixteen from its start.")
    parser.add_argument("--train-action-last2", action="store_true",
                        help="Resume the distinct 8k branch with only the last two Action blocks trainable.")
    parser.add_argument("--no-training-reference", action="store_true",
                        help="Joint-stage continuation without the independent reference or its non-regression loss.")
    parser.add_argument("--full-diagnostics", action="store_true",
                        help="Explicit diagnostic comparison only; late2 training normally retains three metrics.")
    args = parser.parse_args()
    if args.mean_from5k:
        if args.stage != 2 or args.train_action_last2:
            raise ValueError("Mean-from-5k is stage 2, not the legacy last-two-from-8k resume")
        config = paper_con1_mean_from5k_config()
    else:
        config = paper_con1_config(joint=args.stage == 2, late2=args.late2, direct_delta=args.direct_delta,
                                  no_training_reference=args.no_training_reference,
                                  train_action_last2=args.train_action_last2)
    if args.train_action_last2 and not (config.checkpoint_dir / "3000/params/_METADATA").is_file():
        raise ValueError("Prepare the preserved 8k checkpoint in the separate last-two branch before resuming")
    if not 1 <= args.until_step <= config.num_train_steps:
        raise ValueError(f"Stage {args.stage} contains exactly {config.num_train_steps:,} total updates")
    config = dataclasses.replace(
        config, num_train_steps=args.until_step, resume=config.checkpoint_dir.exists(),
        rapr_minimal_diagnostics=(args.late2 or args.mean_from5k) and not args.full_diagnostics)
    path = Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("paper_con1_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(config, diagnose_first_step=True, replay_data_cursor=True)


if __name__ == "__main__":
    main()
