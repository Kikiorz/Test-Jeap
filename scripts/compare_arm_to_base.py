#!/usr/bin/env python3
"""How far did an arm move the weights the released base actually owns?

The probe's `--base-weights` row restores the released checkpoint into every
non-Con1/Con2 parameter. That row is only informative if the arm's fine-tuning
moved those parameters in the first place; if the arm left them at their released
values, `base` and `released base` are the same measurement and one of them is
wasted GPU time.

Reads both checkpoints directly (the released one is in publish layout, the arm
is a step-layout ocdbt/zarr store) and reports the relative L2 change, grouped by
subtree. CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import flax.traverse_util
import numpy as np

from openpi.models import model as _model


def read_step_layout(params_store: str) -> dict:
    """Read an openpi training checkpoint's params without a GPU mesh.

    Its ocdbt store keys are content hashes rather than tree paths, so go through
    orbax instead of the raw zarr reader used by the publish script. The restored
    tree is `{"params": {...}}` with nnx `Variable`s serialised as `.../value`.
    """
    import orbax.checkpoint as ocp

    tree = ocp.PyTreeCheckpointer().restore(str(params_store))
    if isinstance(tree, dict) and set(tree) == {"params"}:
        tree = tree["params"]
    return tree


def normalise_arm_key(key: str) -> str:
    if key.endswith("/value"):
        key = key[: -len("/value")]
    return key


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path,
                   default=Path("/workspace/artifacts/models/jepa_wam_pi05_robotwin_publish/"
                                "pi05_robotwin_clean_20_vjepa_aux/19999/params"))
    p.add_argument("--arm", type=Path,
                   default=Path("/workspace/artifacts/checkpoints/pi05_robotwin_con1_livecross_20k/"
                                "robotwin_a_con1/4499/params"))
    p.add_argument("--buckets", nargs="*",
                   default=["PaliGemma/llm/layers", "action_out_proj", "PaliGemma/img",
                            "PaliGemma/llm/embedding", "action_in_proj", "time_mlp"])
    p.add_argument("--out", type=Path)
    return p.parse_args()


def bucket_of(key: str, buckets: list[str]) -> str:
    for bucket in buckets:
        if key.startswith(bucket):
            return bucket
    return "other"


def main() -> None:
    args = parse()
    print("reading released base ...", flush=True)
    base = _model.restore_params(str(args.base), restore_type=np.ndarray)
    print("reading arm checkpoint ...", flush=True)
    arm = read_step_layout(str(args.arm))

    flat_base = flax.traverse_util.flatten_dict(base, sep="/")
    flat_arm = {normalise_arm_key(k): v
                for k, v in flax.traverse_util.flatten_dict(arm, sep="/").items()}
    shared = sorted(set(flat_base) & set(flat_arm))
    report: dict = {"base": str(args.base), "arm": str(args.arm),
                    "shared_leaves": len(shared), "buckets": {}, "moved": {}}
    totals: dict[str, list] = {}
    for key in shared:
        if "con1" in key or "con2" in key:
            continue
        a = np.asarray(flat_arm[key], np.float32).ravel()
        b = np.asarray(flat_base[key], np.float32).ravel()
        if a.shape != b.shape:
            report.setdefault("shape_mismatch", []).append(key)
            continue
        norm_b = float(np.linalg.norm(b))
        norm_d = float(np.linalg.norm(a - b))
        relative = norm_d / max(norm_b, 1e-12)
        bucket = bucket_of(key, args.buckets)
        totals.setdefault(bucket, []).append((relative, norm_d, norm_b, a.size))
        report["moved"][key] = relative

    for bucket, rows in sorted(totals.items()):
        rel = np.array([r[0] for r in rows])
        report["buckets"][bucket] = {
            "leaves": len(rows),
            "parameters": int(sum(r[3] for r in rows)),
            "relative_l2_median": float(np.median(rel)),
            "relative_l2_max": float(rel.max()),
            "relative_l2_min": float(rel.min()),
            "leaves_moved_over_1e-3": int((rel > 1e-3).sum()),
        }
        print(f"{bucket:26s} leaves={len(rows):4d} median={np.median(rel):.3e} "
              f"max={rel.max():.3e} moved>1e-3={int((rel > 1e-3).sum())}", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
