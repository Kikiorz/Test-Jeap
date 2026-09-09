#!/usr/bin/env python3
"""Four-GPU closed-loop checkpoint evaluation on immutable standard/Plus panels."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from paper_con1_protocol import SUITES, summarize_journals

REPO = Path("/workspace/ts_JEPA_con")
PLUS = Path("/workspace/artifacts/benchmarks/LIBERO-plus")
PLUS_REVISION = "4976dc30028e805ff8094b55501d532c48fec182"
STANDARD_REVISION = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def wait_ready(process, port, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Policy process {process.pid} exited with {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"Policy on port {port} did not become ready")


def run_suite(args, manifest, gpu, suite):
    port = args.port_base + gpu
    server_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                      PYTHONPATH=str(REPO / "src"), HF_HOME="/workspace/.hf_home", HF_HUB_OFFLINE="1",
                      XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.8",
                      OMP_NUM_THREADS="4")
    server_log = args.output / "logs" / f"server_{suite}.log"
    command = [str(REPO / ".venv/bin/python"), "-u", str(REPO / "scripts/serve_policy.py"),
               "--env", "LIBERO", "--host", "127.0.0.1", "--port", str(port)]
    if args.config != "pi05_libero_paper_reference":
        command += ["--rapr-runtime-gate", "1.0"]
    command += ["policy:checkpoint", "--policy.config", args.config, "--policy.dir", str(args.checkpoint)]
    # Never mistake an unrelated service for the child that is still loading.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    with server_log.open("a") as handle:
        process = subprocess.Popen(command, cwd=REPO, env=server_env, stdout=handle, stderr=subprocess.STDOUT)
        try:
            wait_ready(process, port)
            modes = ["plus"] + (["standard"] if args.panel == "monitor" else [])
            for mode in modes:
                journal = args.output / "journals" / f"{mode}_{suite}.jsonl"
                env = dict(os.environ, RUN_ID=args.output.name, HOST="127.0.0.1", PORT=str(port),
                           TASK_SUITE=suite, TASK_START="0", REPLAN_STEPS="5", EVAL_GPU=str(gpu),
                           MUJOCO_EGL_DEVICE_ID=str(gpu),
                           SEED=str(manifest[f"{args.panel}_episode_seed"]), SAVE_VIDEO="0", RETRY_ERRORS="1",
                           LIBERO_PLUS_ROOT=str(PLUS), STANDARD_LIBERO_ROOT=str(REPO / "third_party/libero"),
                           EVAL_PYTHON=str(REPO / "examples/libero/.venv-plus/bin/python"),
                           RESULTS_PATH=str(journal), RESULTS_ROOT=str(args.output),
                           LIBERO_CONFIG_PATH=str(args.output / "runtime" / suite / mode))
                if mode == "plus":
                    env["TASK_IDS"] = " ".join(str(row["task_id"]) for row in manifest[args.panel][suite])
                    env["NUM_TRIALS"] = "1"
                    env["BENCHMARK_REVISION"] = PLUS_REVISION
                    env.pop("TASK_END", None)
                else:
                    env.pop("TASK_IDS", None)
                    env["TASK_END"] = "10"
                    env["NUM_TRIALS"] = str(manifest["monitor_standard_trials_per_task"])
                    env["BENCHMARK_REVISION"] = STANDARD_REVISION
                with (args.output / "logs" / f"{mode}_{suite}.log").open("a") as eval_log:
                    subprocess.run(["bash", str(REPO / "scripts/run_libero_evaluation.sh"), mode],
                                   cwd=REPO, env=env, stdout=eval_log, stderr=subprocess.STDOUT, check=True)
                print(json.dumps({"event": "suite_mode_done", "suite": suite, "mode": mode,
                                  "journal": str(journal)}), flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--panel", choices=("monitor", "final"), default="monitor")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port-base", type=int, default=8700)
    args = parser.parse_args()
    if not (args.checkpoint / "params").is_dir():
        raise FileNotFoundError(args.checkpoint / "params")
    for repo, expected in ((PLUS, PLUS_REVISION), (REPO / "third_party/libero", STANDARD_REVISION)):
        actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise ValueError(f"Benchmark revision mismatch: {repo}: {actual}")
    manifest = json.loads(args.panels.read_text())
    classification = json.loads((PLUS / "libero/libero/benchmark/task_classification.json").read_text())
    classification_hash = hashlib.sha256(json.dumps(classification, sort_keys=True).encode()).hexdigest()
    if manifest["classification_sha256"] != classification_hash:
        raise ValueError("Evaluation task classification changed after panel selection")
    args.output.mkdir(parents=True, exist_ok=True)
    for directory in ("logs", "journals"):
        (args.output / directory).mkdir(exist_ok=True)
    contract = {"config": args.config, "checkpoint": str(args.checkpoint.resolve()), "panel": args.panel,
                "panels": manifest, "plus_revision": PLUS_REVISION, "standard_revision": STANDARD_REVISION}
    contract_path = args.output / "manifest.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Refusing to change an existing evaluation contract")
    atomic_json(contract_path, contract)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_suite, args, manifest, gpu, suite) for gpu, suite in enumerate(SUITES)]
        pending = set(futures)
        while pending:
            finished, pending = concurrent.futures.wait(pending, timeout=30,
                                                       return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                future.result()
            paths = sorted((args.output / "journals").glob("*.jsonl"))
            try:
                live = summarize_journals(paths)
            except json.JSONDecodeError:
                # Writer may be in the middle of one append; retry next poll.
                continue
            atomic_json(args.output / "progress.json", {"unix_time": time.time(), "groups": live,
                        "completed": sum(item["count"] for item in live.values()),
                        "successes": sum(item["success"] for item in live.values())})
            print(json.dumps({"event": "progress", "groups": live}), flush=True)
    groups = summarize_journals(sorted((args.output / "journals").glob("*.jsonl")))
    expected = sum(len(manifest[args.panel][suite]) for suite in SUITES)
    if args.panel == "monitor":
        expected += 40 * manifest["monitor_standard_trials_per_task"]
    actual = sum(group["count"] for group in groups.values())
    errors = sum(group["error"] for group in groups.values())
    report = {"expected_episodes": expected, "completed_episodes": actual, "errors": errors,
              "successes": sum(group["success"] for group in groups.values()), "groups": groups,
              "complete": actual == expected and errors == 0}
    atomic_json(args.output / "summary.json", report)
    print(json.dumps(report), flush=True)
    if not report["complete"]:
        raise RuntimeError("Evaluation incomplete or contains infrastructure errors; inspect journals")


if __name__ == "__main__":
    main()
