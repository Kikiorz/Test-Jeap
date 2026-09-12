#!/usr/bin/env python3
"""Rewrite a step-layout RoboTwin checkpoint into openpi's publish layout.

orbax's composite restore API refuses the nested args this checkpoint needs, so
the parameter tree is read straight out of the ocdbt/zarr store (the keys are
the tree paths, e.g. ``params.PaliGemma.llm.layers.attn.q_einsum.w.value``) and
written back as a plain ``{"params": tree}`` pyTree checkpoint, which is exactly
what ``openpi.models.model.restore_params`` expects.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp
import tensorstore as ts


def read_tree(params_store: str) -> dict:
    base = params_store if "://" in params_store else "file://" + params_store
    kv = ts.KvStore.open({"driver": "ocdbt", "base": base}).result()
    keys = sorted(key.decode() for key in kv.list().result() if key.endswith(b".zarray"))
    tree: dict = {}
    for key in keys:
        path = key[: -len("/.zarray")]
        if not path.startswith("params."):
            raise ValueError(f"unexpected key {path}")
        parts = path[len("params.") :].split(".")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = np.asarray(
            ts.open({"driver": "zarr", "kvstore": {"driver": "ocdbt", "base": base}, "path": path})
            .result()
            .read()
            .result()
        )
    return tree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="step checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="publish-layout directory")
    args = parser.parse_args()

    tree = read_tree(str(args.source / "params"))
    leaves = sum(1 for _ in _walk(tree))
    print(f"read {leaves} leaves", flush=True)

    target = args.output / "params"
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target}")
    ocp.PyTreeCheckpointer().save(target, {"params": tree})
    assets = args.output / "assets"
    if not assets.exists():
        os.symlink((args.source / "assets").resolve(), assets)
    print("WROTE", target)


def _walk(node):
    for value in node.values():
        if isinstance(value, dict):
            yield from _walk(value)
        else:
            yield value


if __name__ == "__main__":
    main()
