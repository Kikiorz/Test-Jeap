"""Mirror ``Sylvest/libero_plus_lerobot`` through the Hugging Face CDN.

``snapshot_download`` resolves the repository revision through the API on every
retry, and that quota (1000 requests / 5 min, shared per token) is already
oversubscribed on this shared box, which is what stalled the first download.
The ``/resolve/`` endpoint is CDN-backed instead, so this script lists the repo
once and then pulls files with curl in parallel.

Usage:
    python scripts/fetch_libero_plus_dataset.py --root /workspace/libero_plus_lerobot
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import pathlib
import subprocess
import time

RESOLVE_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
# The release declares 14,347 episodes of 1,000 per chunk (LeRobot v2.1), which
# lets the fallback below regenerate the file list even from an empty directory.
DEFAULT_EPISODES = 14_347
DEFAULT_CHUNK_SIZE = 1_000


def list_files(repo: str) -> list[str]:
    from huggingface_hub import HfApi

    return HfApi().list_repo_files(repo, repo_type="dataset")


def fallback_list(root: pathlib.Path) -> list[str]:
    """Enumerate from local parquet shards when the API is unreachable."""
    files = ["meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl"]
    episodes = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    pairs = [(parquet.parent.name, parquet.stem) for parquet in episodes]
    # A partial mirror only enumerates the shards it already has, so always add
    # the full declared set as well - otherwise missing parquet files are
    # invisible and the sync reports a complete dataset that is not.
    known = {episode for _, episode in pairs}
    for index in range(DEFAULT_EPISODES):
        episode = f"episode_{index:06d}"
        if episode not in known:
            pairs.append((f"chunk-{index // DEFAULT_CHUNK_SIZE:03d}", episode))
    for chunk, episode in pairs:
        files.append(f"data/{chunk}/{episode}.parquet")
        for view in ("observation.images.front", "observation.images.wrist"):
            files.append(f"videos/{chunk}/{view}/{episode}.mp4")
    return files


def fetch(root: pathlib.Path, relative: str, repo: str, retries: int = 4) -> str:
    target = root / relative
    if target.exists() and target.stat().st_size > 0:
        return "skip"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    url = RESOLVE_URL.format(repo=repo, path=relative)
    for attempt in range(retries):
        result = subprocess.run(
            ["curl", "-L", "-f", "-s", "-S", "--retry", "3", "--retry-delay", "2", "-o", str(tmp), url],
            capture_output=True,
        )
        if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            os.replace(tmp, target)
            return "ok"
        time.sleep(2 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    return "fail"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/libero_plus_lerobot")
    parser.add_argument("--repo", default="Sylvest/libero_plus_lerobot")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    try:
        files = list_files(args.repo)
    except Exception as exc:  # noqa: BLE001 - fall back to whatever is on disk
        print(f"repo listing failed ({type(exc).__name__}: {exc}); using local enumeration", flush=True)
        files = fallback_list(root)

    pending = [name for name in files if not (root / name).exists()]
    print(f"repo has {len(files)} files, {len(pending)} missing in {root}", flush=True)
    if not pending:
        return

    counts: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, root, name, args.repo) for name in pending]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            outcome = future.result()
            counts[outcome] = counts.get(outcome, 0) + 1
            if index % 500 == 0 or index == len(futures):
                print(f"[{index}/{len(futures)}] {counts}", flush=True)
    print("done", counts, flush=True)
    if counts.get("fail"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
