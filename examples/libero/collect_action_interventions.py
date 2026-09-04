#!/usr/bin/env python3
"""Collect same-state, multi-action LIBERO interventions from a frozen policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy


DUMMY_ACTION = [0.0] * 6 + [-1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--init-state-count", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument(
        "--candidate-mode", choices=("policy_samples", "axis_interventions"), default="policy_samples"
    )
    parser.add_argument("--perturbation", type=float, default=0.2)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--stabilization-steps", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--policy-resolution", type=int, default=224)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _make_env(task, resolution: int, seed: int):
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def _images(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    base = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return base, wrist


def _robot_state(obs: dict) -> np.ndarray:
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    axis_angle = np.zeros(3) if denominator < 1e-8 else quat[:3] * 2.0 * np.arccos(quat[3]) / denominator
    return np.concatenate(
        [obs["robot0_eef_pos"], axis_angle, obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


def _policy_element(obs: dict, prompt: str, policy_resolution: int) -> dict:
    base, wrist = _images(obs)
    return {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(base, policy_resolution, policy_resolution)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist, policy_resolution, policy_resolution)
        ),
        "observation/state": _robot_state(obs),
        "prompt": prompt,
    }


def main() -> None:
    args = parse_args()
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    client = websocket_client_policy.WebsocketClientPolicy(
        args.host, args.port, connect_timeout=30.0, inference_timeout=300.0, use_proxy=False
    )

    current_base = []
    current_wrist = []
    future_base = []
    future_wrist = []
    states = []
    action_chunks = []
    prompts = []
    task_indices = []
    episode_indices = []
    candidate_indices = []
    restore_mae = []
    state_id = 0

    try:
        for task_id in args.task_ids:
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            env = _make_env(task, args.resolution, args.seed + task_id)
            try:
                for init_index in range(args.init_state_count):
                    episode_seed = args.seed + 10_000 * task_id + init_index
                    np.random.seed(episode_seed)
                    env.seed(episode_seed)
                    obs = env.reset()
                    obs = env.set_init_state(initial_states[init_index])
                    for _ in range(args.stabilization_steps):
                        obs, _, _, _ = env.step(DUMMY_ACTION)

                    snapshot = np.asarray(env.get_sim_state()).copy()
                    base, wrist = _images(obs)
                    state = _robot_state(obs)
                    element = _policy_element(obs, str(task.language), args.policy_resolution)
                    if args.candidate_mode == "policy_samples":
                        candidates = []
                        for candidate_index in range(args.candidates):
                            policy_seed = episode_seed * 100 + candidate_index
                            action = np.asarray(client.infer(element, seed=policy_seed)["actions"])
                            if action.ndim != 2 or action.shape[0] < args.horizon or action.shape[1] != 7:
                                raise ValueError(f"Unexpected policy action shape {action.shape}")
                            candidates.append(action[: args.horizon].astype(np.float32))
                    else:
                        intervention_basis = np.asarray(
                            [
                                [0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [-1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, -1.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [0.0, 0.0, -1.0],
                            ],
                            dtype=np.float32,
                        )
                        if args.candidates > len(intervention_basis):
                            raise ValueError("axis_interventions supports at most seven candidates")
                        base_action = np.asarray(
                            client.infer(element, seed=episode_seed * 100)["actions"],
                            dtype=np.float32,
                        )[: args.horizon]
                        if base_action.shape != (args.horizon, 7):
                            raise ValueError(f"Unexpected policy action shape {base_action.shape}")
                        candidates = []
                        for offset in intervention_basis[: args.candidates]:
                            action = base_action.copy()
                            action[:, :3] = np.clip(
                                action[:, :3] + args.perturbation * offset[None], -1.0, 1.0
                            )
                            candidates.append(action)

                    for candidate_index, action_chunk in enumerate(candidates):
                        branch_obs = env.set_init_state(snapshot)
                        restored_base, _ = _images(branch_obs)
                        mae = float(
                            np.mean(
                                np.abs(restored_base.astype(np.float32) - base.astype(np.float32))
                            )
                        )
                        for action in action_chunk:
                            branch_obs, _, done, _ = env.step(action.tolist())
                            if done:
                                break
                        branch_base, branch_wrist = _images(branch_obs)

                        current_base.append(base)
                        current_wrist.append(wrist)
                        future_base.append(branch_base)
                        future_wrist.append(branch_wrist)
                        states.append(state)
                        action_chunks.append(action_chunk)
                        prompts.append(str(task.language))
                        task_indices.append(task_id)
                        episode_indices.append(state_id)
                        candidate_indices.append(candidate_index)
                        restore_mae.append(mae)

                    print(
                        json.dumps(
                            {
                                "task": task_id,
                                "init_state": init_index,
                                "state_id": state_id,
                                "branches": args.candidates,
                                "max_restore_mae": max(restore_mae[-args.candidates :]),
                            }
                        ),
                        flush=True,
                    )
                    state_id += 1
            finally:
                env.close()
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    task_indices_array = np.asarray(task_indices, dtype=np.int32)
    episode_indices_array = np.asarray(episode_indices, dtype=np.int32)
    candidate_indices_array = np.asarray(candidate_indices, dtype=np.int32)
    wrong_future_indices = np.empty(len(task_indices_array), dtype=np.int64)
    for index in range(len(wrong_future_indices)):
        same_task = np.flatnonzero(
            (task_indices_array == task_indices_array[index])
            & (episode_indices_array != episode_indices_array[index])
            & (candidate_indices_array == candidate_indices_array[index])
        )
        if not len(same_task):
            raise ValueError("Each task needs at least two states for mismatched futures")
        wrong_future_indices[index] = int(same_task[0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        current_base=np.asarray(current_base, dtype=np.uint8),
        current_wrist=np.asarray(current_wrist, dtype=np.uint8),
        future_base=np.asarray(future_base, dtype=np.uint8),
        future_wrist=np.asarray(future_wrist, dtype=np.uint8),
        states=np.asarray(states, dtype=np.float32),
        action_chunks=np.asarray(action_chunks, dtype=np.float32),
        prompts=np.asarray(prompts),
        task_indices=task_indices_array,
        episode_indices=episode_indices_array,
        # Keep the archive compatible with the JEPA audit/reporting pipeline.
        # Intervention branches do not have dataset frame indices, so the
        # within-state candidate index is the only meaningful local index.
        frame_indices=candidate_indices_array,
        candidate_indices=candidate_indices_array,
        wrong_future_indices=wrong_future_indices,
        restore_mae=np.asarray(restore_mae, dtype=np.float32),
        future_offset=np.asarray(args.horizon, dtype=np.int32),
        candidate_mode=np.asarray(args.candidate_mode),
        perturbation=np.asarray(args.perturbation, dtype=np.float32),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "states": state_id,
                "branches": len(action_chunks),
                "max_restore_mae": max(restore_mae),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
