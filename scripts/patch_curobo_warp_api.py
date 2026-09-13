#!/usr/bin/env python3
"""Patch curobo 0.7.8 for warp-lang >= 1.6.

`curobo/geom/sdf/world_mesh.py` calls `wp.torch.device_from_torch(...)`, the
pre-1.6 spelling. Modern warp-lang dropped the `warp.torch` submodule and exposes
the same helper as `wp.device_from_torch`, so the mesh world-collision path
(`CuroboPlanner.update_point_cloud`) raised `AttributeError: module 'warp' has no
attribute 'torch'` on every reset. `pip install curobo` resolves `warp-lang`
without an upper bound, so this shows up on any fresh install.

Idempotent; pass --revert to undo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

OLD = "wp.torch.device_from_torch("
NEW = "wp.device_from_torch("


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/workspace/robotwin/code/envs/curobo/src/curobo"))
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args()

    source, target = (NEW, OLD) if args.revert else (OLD, NEW)
    touched = 0
    for path in args.root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if source not in text:
            continue
        path.write_text(text.replace(source, target), encoding="utf-8")
        touched += 1
        print(f"patched {path}")
    print(f"{touched} file(s) updated")


if __name__ == "__main__":
    main()
