"""Download the released pi0.5 base checkpoint from the openpi GCS bucket.

The trainer initialises from ``/workspace/models/pi05_base/params``, and that
checkpoint gets deleted once the fine-tune has it in memory (the box only has a
150 GB overlay shared with another job). Re-running this script restores it.

Usage:
    python scripts/fetch_pi05_base.py [--root /workspace/models/pi05_base]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import urllib.request

API = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o"
PREFIX = "checkpoints/pi05_base/"


def list_all() -> list[dict]:
    token, items = None, []
    while True:
        url = f"{API}?prefix={PREFIX}&maxResults=1000&fields=items(name,size),nextPageToken"
        if token:
            url += f"&pageToken={token}"
        payload = json.load(urllib.request.urlopen(url))
        items += payload.get("items", [])
        token = payload.get("nextPageToken")
        if not token:
            break
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/models/pi05_base")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    items = list_all()
    print(f"files {len(items)}", flush=True)

    def fetch(item: dict) -> str:
        name = item["name"]
        dst = os.path.join(args.root, name[len(PREFIX) :])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) == int(item.get("size", 0)):
            return dst
        url = f"https://storage.googleapis.com/openpi-assets/{name}"
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, dst)
                return dst
            except Exception as exc:  # noqa: BLE001 - retry, then surface
                if attempt == 2:
                    print("FAIL", dst, exc, flush=True)
                    raise
        return dst

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        done = list(pool.map(fetch, items))
    print("DOWNLOAD_DONE", len(done), flush=True)


if __name__ == "__main__":
    main()
