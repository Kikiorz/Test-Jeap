"""Does the 5% residual budget cap Con1's accuracy, and is the action path live?

The adapter's correction is capped at ``residual_budget`` times the RMS of the
incoming action hidden state. The alpha sweep already showed the cap - not the
gate - is the binding constraint (the correction RMS stays 0.345 for every
alpha), which raises the obvious question: if the model is saturated at 5%, does
relaxing the cap buy accuracy?

This probe re-evaluates the *same checkpoint* (same parameters, same losses)
under different budgets, so the comparison is paired and needs no retraining.
It also zeroes / shuffles the action conditioning to measure whether the
conditioning path is live end-to-end.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import train
from openpi.models import model as _model
from openpi.training import checkpoints, config as configs, data_loader, sharding


def _path_names(path) -> list[str]:
    names = []
    for entry in path:
        for attribute in ("name", "key", "idx"):
            if hasattr(entry, attribute):
                names.append(str(getattr(entry, attribute)))
                break
        else:
            names.append(str(entry))
    return names


def _zero_correction(params):
    """Base-policy reference: zero the Con1 correction at its output projection.

    Zeroing the output kernel makes the correction identically zero while
    leaving every restored parameter in place, so the run differs from the
    adapter run only by the correction itself.
    """
    def zero(path, leaf):
        names = _path_names(path)
        # Bridged Linen parameters appear as "<layer>/kernel/value".
        if ("con1_cross_attention" in names and len(names) >= 3
                and names[-1] == "value" and names[-2] == "kernel"
                and names[-3] in ("out", "adapter_out")):
            print("ZEROED", "/".join(names), getattr(leaf, "shape", None), flush=True)
            return jnp.zeros_like(leaf)
        return leaf

    return jax.tree_util.tree_map_with_path(zero, params)


def _released_base_params(config, params):
    """Swap every non-Con1/Con2 parameter for the released base checkpoint.

    The arm checkpoints were initialised from the released model and then
    fine-tuned a subset of the action-expert layers, so the plain
    ``--zero-correction`` row still carries that fine-tuning. Substituting the
    released weights back in (the config's own weight loader is the exact one
    used at initialisation) and then silencing the correction yields a true
    released-base row that can be compared on the same batches.
    """
    # `state.params` is an `nnx.State`, and the weight loader flattens its input
    # with `flax.traverse_util.flatten_dict`, which rejects anything that is not a
    # (frozen)dict. Go through the pure-dict view and convert back afterwards.
    pure = params.to_pure_dict()
    spec = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), pure)
    loaded = config.weight_loader.load(spec)

    def pick(path, base_leaf, trained_leaf):
        # Element names look like "con1_cross_attention", not "con1", so match on
        # the joined path rather than on exact elements.
        joined = "/".join(_path_names(path))
        if "con1" in joined or "con2" in joined:
            # Missing from the released checkpoint, and zeroed by the caller.
            return trained_leaf
        return base_leaf

    merged = jax.tree_util.tree_map_with_path(pick, loaded, pure)
    # nnx 0.10 replaces in place and returns None, so the caller gets the same
    # State object back with the shared leaves swapped for the released weights.
    params.replace_by_pure_dict(merged)
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.0],
                        help="Residual budgets to compare; 0 disables the cap.")
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--zero-correction", action="store_true",
                        help="Silence the Con1 correction so the run measures the base policy.")
    parser.add_argument("--base-weights", action="store_true",
                        help="Also restore the released base checkpoint into every non-Con1/Con2 "
                             "parameter, giving the true released-base row in the same harness. "
                             "Implies silencing the correction.")
    parser.add_argument("--dump-param-paths", action="store_true",
                        help="Print the Con1 parameter paths and exit (naming check).")
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()

    jax.config.update("jax_compilation_cache_dir", "/workspace/.cache/con1_jax")
    config = dataclasses.replace(
        configs.get_config(args.config),
        batch_size=args.batch_size, num_workers=0, wandb_enabled=False,
        exp_name=args.exp_name, checkpoint_base_dir="/workspace/artifacts/checkpoints", resume=True,
    )
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=True)
    if not resuming:
        raise ValueError(f"No checkpoint under {config.checkpoint_dir}")
    loader = data_loader.create_data_loader(config, sharding=data_shard, shuffle=True)
    _, init_rng = jax.random.split(jax.random.key(config.seed))
    state, _ = train.init_train_state(config, init_rng, mesh, resume=True)
    jax.block_until_ready(state)
    state = checkpoints.restore_state(manager, state, loader, step=args.checkpoint_step)
    params = state.params
    physical = int(config.model.con1_action_dims)
    if not config.model.con1_action_adapter:
        if not config.model.con1_cross_attention_out_init > 0:
            raise ValueError("This probe needs a Con1 correction path")
    if args.dump_param_paths:
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
            names = [str(getattr(p, "key", getattr(p, "name", p))) for p in path]
            if "con1" in "/".join(names):
                print("PARAM", "/".join(names), getattr(leaf, "shape", None))
        return
    if args.zero_correction:
        params = _zero_correction(params)
    if args.base_weights:
        params = _released_base_params(config, params)
        params = _zero_correction(params)
    # Note: the deployed adapter config leaves the *head* unconditioned; the
    # action-conditioning variants (zero/shuffled chunk) only apply when
    # con1_action_conditioning is set, so they are skipped otherwise.
    conditioned = bool(config.model.con1_action_conditioning)

    def model_def_with_budget(budget):
        if budget == config.model.con1_residual_budget:
            # The requested budget is the one the checkpoint was trained with, so
            # reuse the restored graph instead of building a second model: for the
            # whole-prefix-context variant a second instance does not fit.
            return state.model_def
        cfg = dataclasses.replace(config, model=dataclasses.replace(
            config.model, con1_residual_budget=budget))
        return nnx.graphdef(cfg.model.create(jax.random.key(0)))

    defs = {budget: model_def_with_budget(budget) for budget in args.budgets}

    def flow_of(definition, nparams, observation, x_t, time, action_chunk, action_mask, count, u_t):
        model = nnx.merge(definition, nparams)
        model.eval()
        obs = _model.preprocess_observation(None, observation, train=False)
        context, delta = model._con1_context(obs, action_chunk)
        velocity, aux = model._con1_velocity(obs, x_t, time, context, delta)
        error = jnp.where(action_mask, velocity - u_t, 0.0)
        # The adapter reports mean(correction^2), so this is the correction RMS
        # the budget is actually capping.
        correction_rms = jnp.sqrt(aux["con1_residual_energy"])
        return jnp.square(error).sum() / count, correction_rms, delta

    flow_run = {budget: jax.jit(functools_partial(flow_of, definition),
                                out_shardings=(replicated, replicated, replicated))
                for budget, definition in defs.items()}

    records = []
    iterator = iter(loader)
    for batch_index in range(args.batches):
        observation, actions = next(iterator)
        host = jax.device_get(observation)
        chunk = np.asarray(jax.device_get(actions), np.float32)[..., :physical]
        noise_rng, time_rng = jax.random.split(jax.random.key(args.seed + batch_index))
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        u_t = noise - actions
        valid = np.asarray(host.con1_future_valid, bool)[..., None]
        action_valid = np.concatenate([np.ones_like(valid[:, :1]), valid[:, :-1]], axis=1)
        action_mask = action_valid & (np.arange(actions.shape[-1])[None, None, :] < physical)
        count = jnp.asarray(max(float(action_mask.sum()), 1.0))
        device = dict(mask=jax.device_put(action_mask, replicated), u_t=jax.device_put(u_t, replicated),
                      time=jax.device_put(time, replicated))
        rng = np.random.default_rng(args.seed + batch_index)
        variants = {"true": chunk}
        if conditioned:
            variants["zero"] = np.zeros_like(chunk)
            variants["shuffled"] = chunk[rng.permutation(len(chunk))]
        row = {"batch": batch_index}
        for budget, run in flow_run.items():
            for name, conditioning in variants.items():
                with sharding.set_mesh(mesh):
                    flow, corr_rms, delta = run(params, observation,
                                                jax.device_put(conditioning, replicated),
                                                x_t, device["time"], device["mask"], count, device["u_t"])
                row[f"flow_b{budget}_{name}"] = float(flow)
                if name == "true":
                    row[f"correction_rms_b{budget}"] = float(corr_rms)
        base = row[f"flow_b{args.budgets[0]}_true"]
        for budget in args.budgets:
            row[f"delta_vs_b{args.budgets[0]}_b{budget}"] = (row[f"flow_b{budget}_true"] - base)
        for name in variants:
            if name != "true":
                row[f"delta_{name}"] = row[f"flow_b{args.budgets[0]}_{name}"] - base
        records.append(row)
        print(json.dumps(row), flush=True)

    def mean(key):
        values = np.asarray([r[key] for r in records], dtype=float)
        return float(values.mean()), float(values.std(ddof=1) / max(np.sqrt(len(values)), 1e-9))

    summary = {"config": args.config, "exp_name": args.exp_name, "restored_step": int(state.step),
               "samples": args.batches * args.batch_size, "budgets": args.budgets,
               "flow": {}, "correction_rms": {}, "deltas": {}, "records": records}
    for budget in args.budgets:
        summary["flow"][f"budget_{budget}"] = mean(f"flow_b{budget}_true")
        if f"correction_rms_b{budget}" in records[0]:
            summary["correction_rms"][f"budget_{budget}"] = mean(f"correction_rms_b{budget}")
        key = f"delta_vs_b{args.budgets[0]}_b{budget}"
        summary["deltas"][f"budget_{budget}_vs_{args.budgets[0]}"] = mean(key)
    for name in ("zero", "shuffled"):
        if f"delta_{name}" in records[0]:
            summary["deltas"][f"conditioning_{name}_vs_true"] = mean(f"delta_{name}")
    summary["head_action_conditioning"] = conditioned
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("RESULT", json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=1), flush=True)


def functools_partial(function, definition):
    def wrapped(nparams, observation, action_chunk, x_t, time, action_mask, count, u_t):
        return function(definition, nparams, observation, x_t, time, action_chunk,
                        action_mask, count, u_t)
    return wrapped


if __name__ == "__main__":
    main()
