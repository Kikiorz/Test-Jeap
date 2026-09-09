#!/usr/bin/env python3
"""Fixed episode-held-out, task-balanced offline diagnostics (not task success)."""

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.training import config as config_lib, data_loader, sharding
from openpi.training.paper_metrics import derived_metrics
from run_paper_con1_eval import atomic_json
from paper_con1_protocol import SUITES


def column(dataset, name):
    if isinstance(dataset, data_loader.IndexedDataset):
        return column(dataset._dataset, name)[dataset._indices]
    if hasattr(dataset, "_dataset"):
        return column(dataset._dataset, name)
    return data_loader._task_index_column(dataset, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-task", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--late2", action="store_true")
    parser.add_argument("--direct-delta", action="store_true")
    parser.add_argument("--mean-from5k", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.frames_per_task < 1 or (40 * args.frames_per_task) % args.batch_size:
        raise ValueError("Require a positive, exactly batch-divisible validation panel")
    if args.mean_from5k and args.stage != 2:
        raise ValueError("Mean-from-5k validation is stage 2 only")
    config = (config_lib.paper_con1_mean_from5k_config() if args.mean_from5k else
              config_lib.paper_con1_config(joint=args.stage == 2, late2=args.late2, direct_delta=args.direct_delta))
    factory = dataclasses.replace(config.data, episode_split="validation")
    data_config = factory.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    tasks, episodes, frames = (column(dataset, name) for name in ("task_index", "episode_index", "frame_index"))
    selection = []
    for task in range(40):
        candidates = np.flatnonzero(tasks == task)
        if len(candidates) < args.frames_per_task:
            raise ValueError(f"Insufficient held-out rows for task {task}")
        rng = np.random.default_rng(np.random.SeedSequence([917, task]))
        selection.extend(rng.choice(candidates, args.frames_per_task, replace=False).tolist())
    selection = np.array(sorted(selection))
    row_identity = [{"task": int(tasks[i]), "episode": int(episodes[i]), "frame": int(frames[i])}
                    for i in selection]
    if args.plan_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, {"plan_only": True, "complete": False, "selection_seed": 917,
                                 "frames": len(selection), "rows": row_identity,
                                 "task_counts": {str(task): int(np.sum(tasks[selection] == task)) for task in range(40)},
                                 "note": "Data selection audit only; no model validation performed."})
        print(json.dumps({"frames": len(selection), "tasks": len(set(tasks[selection])), "output": str(args.output)}))
        return
    dataset = data_loader.IndexedDataset(dataset, selection)
    dataset = data_loader.transform_dataset(dataset, data_config)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = data_loader.DataLoaderImpl(data_config, data_loader.TorchDataLoader(
        dataset, args.batch_size, sharding=data_sharding, shuffle=False,
        num_batches=len(selection) // args.batch_size, num_workers=0))
    model = config.model.load(model_lib.restore_params(args.checkpoint / "params", restore_type=np.ndarray))
    model.eval()
    graph, params = nnx.split(model)
    params = jax.device_put(params, sharding.fsdp_sharding(params, mesh))
    trainer_path = Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("paper_validation_train", trainer_path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    reference_config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, rapr_train_action_expert=True, rapr_nonregression_weight=1.0),
        rapr_reference_params=config_lib.PAPER_CON1_BASE_PARAMS)
    reference, _ = trainer.init_fixed_reference(reference_config, mesh)

    @jax.jit
    def evaluate(params, reference, obs, actions, rng):
        candidate = nnx.merge(graph, params)
        baseline = nnx.merge(reference.model_def, reference.params)
        velocity = baseline.reference_training_velocity(rng, obs, actions, train=False)
        flow, auxiliary, _, metrics = candidate.compute_all_loss_components(
            rng, obs, actions, train=False, reference_velocity=velocity)
        metrics = dict(metrics, flow_loss=flow.mean(-1), vjepa_loss=auxiliary)
        noise = jax.random.normal(jax.random.fold_in(rng, 1), actions.shape)
        predicted = candidate.sample_actions(rng, obs, num_steps=10, noise=noise)
        original = baseline.sample_actions(rng, obs, num_steps=10, noise=noise)
        for name, section in (("action7", slice(0, 7)), ("translation", slice(0, 3)),
                              ("rotation", slice(3, 6)), ("gripper", slice(6, 7))):
            metrics[f"chunk_{name}_mse"] = jnp.square(predicted[..., section] - actions[..., section]).mean((-1, -2))
            metrics[f"reference_chunk_{name}_mse"] = jnp.square(original[..., section] - actions[..., section]).mean((-1, -2))
        metrics["chunk_reference_action7_rms"] = jnp.sqrt(jnp.square(predicted[..., :7] - original[..., :7]).mean((-1, -2)))
        return metrics

    values = []
    for index, (obs, actions) in enumerate(loader):
        with sharding.set_mesh(mesh):
            metrics = jax.device_get(evaluate(params, reference, obs, actions, jax.random.fold_in(jax.random.key(917), index)))
        if not all(np.isfinite(value).all() for value in metrics.values()):
            raise FloatingPointError(f"Non-finite validation metrics in batch {index}")
        for offset, task in enumerate(np.asarray(obs.task_index)):
            values.append({"task": int(task), **{key: float(value[offset]) for key, value in metrics.items()}})
        print(json.dumps({"event": "validation_batch", "completed_frames": len(values), "total": len(selection)}), flush=True)
    if len(values) != len(selection):
        raise AssertionError("Validation dropped rows")
    groups = {}
    for name, task_set in [("all", set(range(40)))] + [(suite, set(range(i * 10, (i + 1) * 10))) for i, suite in enumerate(SUITES)] + [(f"task_{i}", {i}) for i in range(40)]:
        selected = [row for row in values if row["task"] in task_set]
        groups[name] = {"count": len(selected), **{key: float(np.mean([row[key] for row in selected]))
                                                    for key in values[0] if key != "task"}}
        # Aggregate energy ratio is more stable than averaging tiny-denominator per-example ratios.
        groups[name]["delta_aggregate_nmse"] = groups[name]["rapr_delta_mse"] / max(groups[name]["rapr_delta_zero_mse"], 1e-30)
        groups[name] = derived_metrics(groups[name])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, {"checkpoint": str(args.checkpoint), "stage": args.stage, "complete": True,
                             "prediction_feature_reduction": config.model.rapr_prediction_reduction,
                             "transition_target_mode": data_config.transition_target_mode,
                             "note": "Normalized offline errors, not closed-loop success; baseline may have seen these demonstrations before Con1.",
                             "selection_seed": 917, "rows": row_identity, "groups": groups})


if __name__ == "__main__":
    main()
