#!/usr/bin/env python3
"""Can a head-only checkpoint be grafted into the joint model's delta head?

A head-only run trains `AnchoredDeltaHead` as a bare Linen module, while the joint
model bridges it with `nnx_bridge.ToNNX`, so the parameter paths are not
obviously the same. This compares the two key sets on CPU: if every head-only key
`k` appears in the joint tree as `con1_delta_head/k` (and the shapes agree), a warm
start is a plain copy and no loader surgery is needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import flax.nnx as nnx
import flax.traverse_util
import jax
import numpy as np
from flax import serialization

from openpi.training import config as configs


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("/workspace/artifacts/con2/robotwin_head_warmstart.msgpack"))
    p.add_argument("--config", default="pi05_robotwin_con1_livecross_20k")
    p.add_argument("--prefix", default="con1_delta_head")
    p.add_argument("--apply", action="store_true",
                   help="Also run the probe's _graft_head against an eval-shape state, so the "
                        "graft code itself is exercised and not just the key comparison.")
    return p.parse_args()


def key_of(path) -> str:
    parts = []
    for entry in path:
        parts.append(str(getattr(entry, "key", getattr(entry, "idx", entry))))
    return "/".join(parts)


def main() -> None:
    args = parse()
    # msgpack_restore rebuilds whatever tree was written, so both the bare
    # {"<module>": ...} tree and the {"params", "opt_state", "step"} wrapper work.
    blob = serialization.msgpack_restore(args.checkpoint.read_bytes())
    head = blob["params"] if isinstance(blob, dict) and "params" in blob else blob
    head_flat = jax.tree_util.tree_flatten_with_path(head)[0]

    config = configs.get_config(args.config)
    model = nnx.eval_shape(config.model.create, jax.random.key(0))
    state = nnx.state(model)
    joint_flat = jax.tree_util.tree_flatten_with_path(state.to_pure_dict())[0]
    joint_keys = {
        key_of(path): leaf
        for path, leaf in joint_flat
        if args.prefix in key_of(path)
    }

    print(f"head-only leaves : {len(head_flat)}  (step {blob.get('step')})")
    print(f"joint {args.prefix} leaves: {len(joint_keys)}")

    prefixed_exact = 0
    missing: list[str] = []
    shape_mismatch: list[tuple[str, tuple, tuple]] = []
    for path, leaf in head_flat:
        key = key_of(path)
        match = None
        for candidate in (f"{args.prefix}/{key}", f"{args.prefix}/params/{key}"):
            if candidate in joint_keys:
                match = candidate
                break
        if match is None:
            missing.append(key)
            continue
        joint_shape = tuple(np.shape(joint_keys[match]))
        if joint_shape != tuple(np.shape(leaf)):
            shape_mismatch.append((key, tuple(np.shape(leaf)), joint_shape))
        else:
            prefixed_exact += 1

    print(f"exact matches    : {prefixed_exact}/{len(head_flat)}")
    if missing:
        print(f"missing ({len(missing)}), e.g.")
        for key in missing[:6]:
            print("   ", key)
        print("   joint keys, e.g.")
        for key in sorted(joint_keys)[:6]:
            print("   ", key)
    if shape_mismatch:
        print(f"shape mismatches ({len(shape_mismatch)}), e.g. {shape_mismatch[:3]}")
    if not missing and not shape_mismatch:
        print("OK: every head-only leaf maps onto a joint delta-head leaf by prefix alone")
    else:
        raise SystemExit(1)

    if args.apply:
        from probe_con1_budget_and_conditioning import _graft_head

        state = nnx.state(model)
        # _graft_head raises unless it rewrites every leaf it read, so a clean call
        # is already the assertion; this counts the subtree afterwards as a second
        # check that nothing was dropped. Key *formatting* changes once real arrays
        # replace the eval-shape placeholders, so match on names, not on strings.
        _graft_head(state, args.checkpoint)
        after = flax.traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
        head_leaves = [k for k in after if any(name in k for name in ("anchor_in", "chunk_expand",
                                                                      "delta_out", "cross_attention"))]
        print(f"graft applied; {len(head_leaves)} head leaves present afterwards "
              f"(expected {len(head_flat)})")
        if len(head_leaves) != len(head_flat):
            raise SystemExit("graft changed the number of head leaves")


if __name__ == "__main__":
    main()
