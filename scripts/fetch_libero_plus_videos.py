"""Fetch missing LIBERO-Plus videos straight from the Hugging Face CDN.

``snapshot_download`` re-resolves the repository revision at the start of every
retry, which counts against the 1000-requests-per-5-minutes API quota that is
shared per token. When that quota is exhausted the whole round aborts before a
single byte moves, which is what stalled the download here.

The ``/resolve/`` endpoint is served by the CDN rather than the rate-limited
API, so this script enumerates the expected files from the local parquet shards
and pulls only the missing ones with curl.

Usage:
    python scripts/fetch_libero_plus_videos.py [--workers 12] [--repo ...]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import pathlib
import subprocess
import time

RESOLVE_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
VIEWS = ("observation.images.front", "observation.images.wrist")


def expected_files(root: pathlib.Path):
    """Yield (local_path, repo_relative_path) for every video the dataset declares."""
    for chunk_dir in sorted((root / "data").glob("chunk-*")):
        for parquet in sorted(chunk_dir.glob("episode_*.parquet")):
            episode = parquet.stem
            for view in VIEWS:
                local = root / "videos" / chunk_dir.name / view / f"{episode}.mp4"
                relative = f"videos/{chunk_dir.name}/{view}/{episode}.mp4"
                yield local, relative


def fetch(local: pathlib.Path, relative: str, repo: str, retries: int = 4) -> str:
    if local.exists() and local.stat().st_size > 0:
        return "skip"
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(local.name + ".part")
    url = RESOLVE_URL.format(repo=repo, path=relative)
    for attempt in range(retries):
        result = subprocess.run(
            ["curl", "-L", "-f", "-s", "-S", "--retry", "3", "--retry-delay", "2", "-o", str(tmp), url],
            capture_output=True,
        )
        if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            os.replace(tmp, local)
            return "ok"
        time.sleep(2 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    return "fail"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/data/libero_plus_lerobot")
    parser.add_argument("--repo", default="Sylvest/libero_plus_lerobot")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    pending = [(local, relative) for local, relative in expected_files(root) if not local.exists()]
    print(f"missing {len(pending)} of the expected videos", flush=True)
    if not pending:
        return

    counts: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, local, relative, args.repo) for local, relative in pending]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            outcome = future.result()
            counts[outcome] = counts.get(outcome, 0) + 1
            if index % 200 == 0 or index == len(futures):
                print(f"[{index}/{len(futures)}] {counts}", flush=True)
    print("done", counts, flush=True)


if __name__ == "__main__":
    main()
