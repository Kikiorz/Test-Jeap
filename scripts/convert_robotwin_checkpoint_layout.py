#!/usr/bin/env python3
"""Convert a training-layout checkpoint into the publish layout openpi expects.

The released RoboTwin checkpoint is a *step* checkpoint (``_CHECKPOINT_METADATA``
holding ``assets`` / ``params`` / ``train_state`` items). openpi's inference and
weight-loading paths call ``restore_params(<dir>/params)``, which needs
``params/_CHECKPOINT_METADATA``. This restores the ``params`` item from the step
checkpoint and re-saves it as a plain ``{"params": tree}`` pyTree checkpoint,
and links the assets directory next to it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import orbax.checkpoint as ocp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    (args.output).mkdir(parents=True, exist_ok=True)
    with ocp.Checkpointer(ocp.StandardCheckpointHandler()) as ckptr:
        restored = ckptr.restore(
            args.source,
            args=ocp.args.StandardRestore(item={"params": ocp.args.PyTreeRestore()}),
        )
    params = restored["params"]
    target = args.output / "params"
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target}")
    ocp.PyTreeCheckpointer().save(target, {"params": params})

    assets = args.output / "assets"
    if not assets.exists():
        os.symlink((args.source / "assets").resolve(), assets)
    print("WROTE", target)
    print("ASSETS", assets, "->", (args.source / "assets").resolve())


if __name__ == "__main__":
    main()
