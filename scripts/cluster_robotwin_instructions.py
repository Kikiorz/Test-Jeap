#!/usr/bin/env python3
"""How many distinct tasks does the RoboTwin 2.0 release actually contain?

`meta/info.json` reports `total_tasks: 2410`, but that counts distinct
instruction *strings*, and the strings are paraphrases of the same behaviour
("Pick the bottle with ridges near base head-up using the left arm" vs "Use the
left arm to pick the hand-sized soda bottle up and keep it head-up"). The release
exposes no task identity anywhere, so estimate it: TF-IDF over the instructions
and greedy cosine clustering, reporting how many families survive at several
thresholds and what the largest ones are about.

Pure numpy; sklearn is not installed on the training box.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path

import numpy as np

TOKEN = re.compile(r"[a-z]+")
STOP = set("""a an the to of in on at with and or for from into onto by using use used
keep hold make makes then first next it its this that left right arm arms hand hands
correct proper appropriate suitable careful carefully please your you own both side sides
up down out over around near between above below center middle top bottom place put pick
raise grab take move pass give hand transfer drop lay set stand""".split())


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path,
                   default=Path("/workspace/robotwin2/RoboTwin_v21/meta/episodes.jsonl"))
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--top", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse()
    rows = [json.loads(line) for line in args.episodes.read_text().splitlines() if line.strip()]
    texts = [" ".join(r["tasks"]).lower() for r in rows]

    # Document frequency over content tokens only.
    docs_tokens = [[t for t in TOKEN.findall(t) if t not in STOP and len(t) > 2] for t in texts]
    df: collections.Counter = collections.Counter()
    for tokens in docs_tokens:
        df.update(set(tokens))
    vocabulary = sorted(df)
    index = {term: i for i, term in enumerate(vocabulary)}
    n_docs = len(docs_tokens)

    vectors = np.zeros((n_docs, len(vocabulary)), np.float32)
    for row, tokens in enumerate(docs_tokens):
        counts = collections.Counter(tokens)
        for term, count in counts.items():
            # Smoothed idf; the +1 keeps terms that appear in every document.
            vectors[row, index[term]] = (1.0 + math.log(count)) * math.log((1 + n_docs) / (1 + df[term])) + 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, 1e-9)

    print(f"instructions: {n_docs}   vocabulary: {len(vocabulary)}")
    print()
    print("greedy cosine clusters at several thresholds")
    print(f"{'threshold':>10} {'clusters':>9} {'largest':>8} {'median size':>12}")
    for threshold in args.thresholds:
        centroids: list[np.ndarray] = []
        sizes: list[int] = []
        members: list[list[int]] = []
        for row in range(n_docs):
            if centroids:
                similarities = np.asarray([float(vectors[row] @ c) for c in centroids])
                best = int(np.argmax(similarities))
                if similarities[best] >= threshold:
                    sizes[best] += 1
                    members[best].append(row)
                    # Running mean keeps the centroid cheap to update.
                    centroids[best] = centroids[best] + (vectors[row] - centroids[best]) / sizes[best]
                    continue
            centroids.append(vectors[row].copy())
            sizes.append(1)
            members.append([row])
        order = np.argsort(sizes)[::-1]
        print(f"{threshold:>10.2f} {len(centroids):>9} {max(sizes):>8} "
              f"{int(np.median(sizes)):>12}")
        if abs(threshold - 0.5) < 1e-9:
            print()
            print(f"largest clusters at {threshold}:")
            for rank in order[:args.top]:
                idx = int(rank)
                terms = collections.Counter()
                for row in members[idx]:
                    terms.update(docs_tokens[row])
                top_terms = ", ".join(t for t, _ in terms.most_common(6))
                example = texts[members[idx][0]][:70]
                print(f"  {sizes[idx]:5d}  [{top_terms}]")
                print(f"         e.g. {example}")


if __name__ == "__main__":
    main()
