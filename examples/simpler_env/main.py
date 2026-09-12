"""SimplerEnv (WidowX / Bridge) evaluation for openpi pi0.5 policies.

Four tasks, 24 trials each by default — the protocol the SimplerEnv repo uses
for its WidowX (Bridge) results (``scripts/octo_bridge.sh`` runs
``--obj-episode-range 0 24``).

This script runs inside the SimplerEnv environment; the policy is served
out-of-process by ``scripts/serve_policy.py`` and queried over websocket, which
keeps the JAX/openpi dependencies out of the simulator environment.

Action conventions follow SimplerEnv's own ``widowx_bridge`` policy setup:
world vector straight through, rotation deltas converted from rpy to an
axis-angle, and the gripper binarised to +1 (open) / -1 (close).
"""

from __future__ import annotations

import argparse

import json
import pathlib
import time

import numpy as np
import simpler_env
from transforms3d.euler import euler2axangle

from openpi_client import image_tools
from openpi_client import websocket_client_policy

TASKS = {
    "carrot_on_plate": "widowx_carrot_on_plate",
    "spoon_on_towel": "widowx_spoon_on_towel",
    "eggplant_in_basket": "widowx_put_eggplant_in_basket",
    "stack_cube": "widowx_stack_cube",
}


def _quat_to_matrix(quat) -> np.ndarray:
    """wxyz quaternion -> 3x3 rotation matrix."""
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return float(roll), float(pitch), float(yaw)


def bridge_state(obs: dict) -> np.ndarray:
    """8-dim Bridge state: [x, y, z, roll, pitch, yaw, 0, gripper_openness].

    Bridge stores the end-effector pose **relative to the robot base**, while
    SimplerEnv reports the world-frame ``tcp_pose`` (z is the table height, ~1 m,
    versus a Bridge z mean of 0.064). Feeding the world pose costs ~18 standard
    deviations on z and flips the roll, so the base transform is mandatory.
    """
    tcp = np.asarray(obs["extra"]["tcp_pose"], dtype=np.float64).reshape(-1)  # xyz + quat(wxyz)
    base = np.asarray(obs["agent"]["base_pose"], dtype=np.float64).reshape(-1)  # xyz + quat(wxyz)
    base_rotation = _quat_to_matrix(base[3:7])
    relative_position = base_rotation.T @ (tcp[:3] - base[:3])
    relative_rotation = base_rotation.T @ _quat_to_matrix(tcp[3:7])
    roll, pitch, yaw = _matrix_to_rpy(relative_rotation)
    qpos = np.asarray(obs["agent"]["qpos"], dtype=np.float64).reshape(-1)
    # The two finger joints mirror each other; openness in [0, 1] with 1 = open.
    openness = float(np.clip(1.0 - np.mean(np.abs(qpos[-2:])) / 0.04, 0.0, 1.0))
    return np.concatenate([relative_position, [roll, pitch, yaw], [0.0], [openness]]).astype(np.float32)


def to_env_action(action: np.ndarray, *, open_threshold: float, invert_gripper: bool = False) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    world = action[:3]
    rot_ax, rot_angle = euler2axangle(action[3], action[4], action[5])
    gripper = 2.0 * (action[6] > open_threshold) - 1.0
    if invert_gripper:
        gripper = -gripper
    return np.concatenate([world, rot_ax * rot_angle, [gripper]]).astype(np.float64)


def rollout(env, policy, *, replan_steps: int, open_threshold: float, invert_gripper: bool = False,
            use_wrist_image: bool = False, max_steps: int = 120):
    obs, _ = env.reset()
    instruction = env.get_language_instruction()
    done = False
    success = False
    steps = 0
    while not done and steps < max_steps:
        request = {
            "observation/image": np.asarray(obs["image"]["3rd_view_camera"]["rgb"], dtype=np.uint8),
            "observation/state": bridge_state(obs),
            "prompt": instruction,
        }
        if use_wrist_image:
            request["observation/wrist_image"] = np.asarray(
                obs["image"]["base_camera"]["rgb"], dtype=np.uint8)
        chunk = np.asarray(policy.infer(request)["actions"], dtype=np.float64)
        for k in range(min(replan_steps, chunk.shape[0])):
            action = to_env_action(chunk[k], open_threshold=open_threshold, invert_gripper=invert_gripper)
            obs, _reward, success, truncated, _info = env.step(action)
            steps += 1
            done = bool(truncated) or bool(success) or steps >= max_steps
            if done:
                break
        if chunk.shape[0] == 0:
            break
    return bool(success), steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="all", choices=["all", *TASKS.keys()])
    parser.add_argument("--n-trajs", type=int, default=24)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--open-threshold", type=float, default=0.5)
    parser.add_argument("--use-wrist-image", action="store_true",
                        help="Also send SimplerEnv's wrist camera as observation/wrist_image.")
    parser.add_argument("--invert-gripper", action="store_true",
                        help="Treat the model's gripper channel as 'closedness' instead of 'openness'.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=pathlib.Path, default=pathlib.Path("data/simpler_env_eval"))
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"server metadata: {policy.get_server_metadata()}")

    chosen = list(TASKS.items()) if args.task == "all" else [(args.task, TASKS[args.task])]
    summary = {}
    for label, env_name in chosen:
        env = simpler_env.make(env_name)
        results = []
        for trial in range(args.n_trajs):
            start = time.time()
            success, steps = rollout(
                env, policy, replan_steps=args.replan_steps, open_threshold=args.open_threshold,
                invert_gripper=args.invert_gripper, use_wrist_image=args.use_wrist_image,
            )
            results.append(success)
            print(f"[{label}] trial {trial + 1}/{args.n_trajs} success={success} steps={steps} "
                  f"({time.time() - start:.1f}s)", flush=True)
        env.close()
        rate = float(np.mean(results))
        summary[label] = {"n": len(results), "successes": int(np.sum(results)), "success_rate": rate}
        print(f"[{label}] {int(np.sum(results))}/{len(results)} = {rate:.1%}", flush=True)

    out = args.log_dir / f"simpler_env_{args.task}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                               "summary": summary}, indent=2) + "\n")
    print("WROTE", out)


if __name__ == "__main__":
    main()
