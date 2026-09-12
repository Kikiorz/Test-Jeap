#!/usr/bin/env python3
"""CPU check for the probe's `--base-weights` path, run before the GPU probe.

`init_train_state(resume=False)` already applies the config's weight loader, so
re-applying it to an arm checkpoint must reproduce the initialised values on
every non-Con1/Con2 leaf. That gives three things to verify without a GPU:

1. the released checkpoint supplies a value for every non-Con1/Con2 leaf, i.e.
   substituting it back in is well defined for an arm checkpoint;
2. the resulting tree has the same structure and dtypes as the input, and no
   `jax.ShapeDtypeStruct` leaks through (a leak would break the probe's jit);
3. the Con1/Con2 leaves are passed through untouched, so only the correction is
   silenced and the rest of the architecture stays intact.
"""
from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np

import train
from openpi.training import config as configs, sharding

from probe_con1_budget_and_conditioning import _released_base_params


def main() -> None:
    jax.config.update("jax_platform_name", "cpu")
    base = configs.get_config("pi05_robotwin_con1_livecross_20k")
    config = dataclasses.replace(
        base, fsdp_devices=1, batch_size=1, num_workers=0, wandb_enabled=False,
        exp_name="released_base_check", resume=False)
    mesh = sharding.make_mesh(config.fsdp_devices)
    _, rng = jax.random.split(jax.random.key(config.seed))
    state, _ = train.init_train_state(config, rng, mesh, resume=False)
    params = state.params

    swapped = _released_base_params(config, params)

    # `state.params` is an nnx.State; flatten its pure-dict view instead.
    flat_params = flax.traverse_util.flatten_dict(params.to_pure_dict(), sep="/")
    flat_swapped = flax.traverse_util.flatten_dict(swapped.to_pure_dict(), sep="/")

    problems: list[str] = []
    if set(flat_params) != set(flat_swapped):
        missing = set(flat_params) ^ set(flat_swapped)
        problems.append(f"structure mismatch: {sorted(missing)[:5]}")

    leaks = [(k, type(v).__name__) for k, v in flat_swapped.items()
             if isinstance(v, jax.ShapeDtypeStruct)]
    if leaks:
        problems.append(f"{len(leaks)} ShapeDtypeStruct leaks, e.g. {leaks[:3]}")

    con1_like = 0
    reproduced = 0
    changed = 0
    for key, original in flat_params.items():
        replacement = flat_swapped.get(key)
        if replacement is None:
            continue
        is_con1 = "con1" in key or "con2" in key
        if is_con1:
            con1_like += 1
            if not np.array_equal(np.asarray(original), np.asarray(replacement)):
                changed += 1
            continue
        a = np.asarray(original)
        b = np.asarray(replacement)
        if a.shape != b.shape:
            problems.append(f"{key}: shape {a.shape} -> {b.shape}")
            continue
        if a.dtype != b.dtype:
            problems.append(f"{key}: dtype {a.dtype} -> {b.dtype}")
            continue
        if np.array_equal(a, b):
            reproduced += 1
        else:
            problems.append(f"{key}: value differs from the initialised released weight")

    print(f"leaves total          : {len(flat_params)}")
    print(f"non-Con1/Con2 leaves  : {reproduced} reproduced, "
          f"{len(problems)} problems")
    print(f"Con1/Con2 leaves      : {con1_like} (pass-through, {changed} differ from init)")
    print(f"ShapeDtypeStruct leaks: {len(leaks)}")
    if problems:
        print("FAIL")
        for line in problems[:10]:
            print("  ", line)
        raise SystemExit(1)
    if leaks:
        raise SystemExit(1)

    # The probe hands this tree to nnx.merge and then to jitted code, so the merge
    # and a first forward have to survive the mixed dtypes (released float32 over
    # the arm's bfloat16 Con1 leaves).
    model = nnx.merge(state.model_def, swapped)
    model.eval()
    r_tokens = jnp.zeros((1, config.model.vjepa_num_queries, 2048), jnp.float32)
    current = jnp.zeros((1, config.model.con1_latent_dim), jnp.float32)
    out = model.con1_delta_head(r_tokens, current)
    delta = out["delta"]
    print(f"merge+forward  : delta {tuple(delta.shape)} {delta.dtype}")
    if not bool(jnp.isfinite(delta).all()):
        raise SystemExit("head produced non-finite values")
    if delta.shape != (1, config.model.action_horizon, config.model.con1_latent_dim):
        raise SystemExit(f"unexpected head output shape {delta.shape}")
    print("OK: --base-weights restores the released checkpoint on every shared leaf, "
          "merges and runs")


if __name__ == "__main__":
    main()
