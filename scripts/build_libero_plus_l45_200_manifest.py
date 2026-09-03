#!/usr/bin/env python3
"""Build a sealed, category-balanced LIBERO-Plus L4/L5 evaluation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATEGORIES = (
    "Background Textures",
    "Camera Viewpoints",
    "Language Instructions",
    "Light Conditions",
    "Objects Layout",
    "Robot Initial States",
    "Sensor Noise",
)
DIFFICULTIES = (4, 5)


def _rank(seed: int, *parts: object) -> str:
    payload = "|".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode()).hexdigest()


def _allocate_quotas(classification: dict, seed: int, difficulty: int) -> dict[tuple[str, str], int]:
    capacities = {
        (suite, category): sum(
            row.get("category") == category and row.get("difficulty_level") == difficulty
            for row in classification[suite]
        )
        for suite in SUITES
        for category in CATEGORIES
    }
    quotas = {key: min(3, capacity) for key, capacity in capacities.items()}

    # 100 samples over seven categories gives two categories with 15 and five
    # with 14. Rotate the two extras deterministically between L4 and L5.
    extra_categories = sorted(CATEGORIES, key=lambda category: _rank(seed, "column", difficulty, category))[:2]
    column_targets = {category: 14 + int(category in extra_categories) for category in CATEGORIES}

    row_needs = {suite: 25 - sum(quotas[(suite, category)] for category in CATEGORIES) for suite in SUITES}
    column_needs = {
        category: column_targets[category] - sum(quotas[(suite, category)] for suite in SUITES)
        for category in CATEGORIES
    }

    # Solve the remaining small transportation problem with integral max flow.
    # A local greedy allocator can strand a sparse cell (e.g. Spatial/L5
    # background); augmenting paths allow earlier choices to be rerouted.
    source = ("source",)
    sink = ("sink",)
    adjacency: dict[tuple, list[tuple]] = {}
    residual: dict[tuple[tuple, tuple], int] = {}
    original: dict[tuple[tuple, tuple], int] = {}

    def add_edge(left: tuple, right: tuple, capacity: int) -> None:
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
        residual[(left, right)] = capacity
        residual[(right, left)] = 0
        original[(left, right)] = capacity

    ordered_suites = sorted(SUITES, key=lambda suite: _rank(seed, "suite", difficulty, suite))
    for suite in ordered_suites:
        add_edge(source, ("suite", suite), row_needs[suite])
        ordered_categories = sorted(
            CATEGORIES, key=lambda category: _rank(seed, "cell", difficulty, suite, category)
        )
        for category in ordered_categories:
            add_edge(
                ("suite", suite),
                ("category", category),
                capacities[(suite, category)] - quotas[(suite, category)],
            )
    for category in CATEGORIES:
        add_edge(("category", category), sink, column_needs[category])

    flow = 0
    target_flow = sum(row_needs.values())
    while flow < target_flow:
        parent: dict[tuple, tuple | None] = {source: None}
        queue = [source]
        for node in queue:
            for neighbor in adjacency.get(node, []):
                if neighbor not in parent and residual.get((node, neighbor), 0) > 0:
                    parent[neighbor] = node
                    queue.append(neighbor)
                    if neighbor == sink:
                        break
            if sink in parent:
                break
        if sink not in parent:
            raise ValueError(
                f"Cannot satisfy balanced quotas for difficulty {difficulty}: "
                f"{row_needs=}, {column_needs=}, {capacities=}"
            )
        amount = target_flow - flow
        node = sink
        while parent[node] is not None:
            amount = min(amount, residual[(parent[node], node)])
            node = parent[node]
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            residual[(previous, node)] -= amount
            residual[(node, previous)] += amount
            node = previous
        flow += amount

    for suite in SUITES:
        for category in CATEGORIES:
            edge = (("suite", suite), ("category", category))
            quotas[(suite, category)] += original[edge] - residual[edge]

    actual_columns = {
        category: sum(quotas[(suite, category)] for suite in SUITES) for category in CATEGORIES
    }
    if actual_columns != column_targets:
        raise ValueError(f"Column quota mismatch for difficulty {difficulty}: {actual_columns} != {column_targets}")
    return quotas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    classification = json.loads(args.classification.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    total_by_difficulty = {difficulty: 0 for difficulty in DIFFICULTIES}
    selected_keys: set[tuple[str, int]] = set()
    allocated = {
        difficulty: _allocate_quotas(classification, args.seed, difficulty) for difficulty in DIFFICULTIES
    }

    for suite in SUITES:
        rows = classification[suite]
        selected: list[dict] = []
        quotas: dict[str, dict[str, int]] = {}
        candidate_counts: dict[str, dict[str, int]] = {}
        for difficulty in DIFFICULTIES:
            quotas[str(difficulty)] = {}
            candidate_counts[str(difficulty)] = {}
            for category in CATEGORIES:
                quota = allocated[difficulty][(suite, category)]
                candidates = [
                    (task_id, row)
                    for task_id, row in enumerate(rows)
                    if row.get("category") == category and row.get("difficulty_level") == difficulty
                ]
                candidates.sort(
                    key=lambda item: _rank(
                        args.seed, suite, difficulty, category, item[0], item[1]["name"]
                    )
                )
                if len(candidates) < quota:
                    raise ValueError(
                        f"Insufficient candidates for {suite=} {difficulty=} {category=}: "
                        f"need {quota}, found {len(candidates)}"
                    )
                quotas[str(difficulty)][category] = quota
                candidate_counts[str(difficulty)][category] = len(candidates)
                chosen = candidates[:quota]
                selected.extend(chosen)
                total_by_difficulty[difficulty] += len(chosen)

        selected.sort(key=lambda item: _rank(args.seed, "order", suite, item[0]))
        # task_classification.json exposes a human-facing 1-based `id`, while
        # LIBERO's evaluator indexes the task list from zero.  The sealed
        # manifest must therefore use the classification array index.
        task_ids = [task_id for task_id, _ in selected]
        if len(task_ids) != 50 or len(set(task_ids)) != 50:
            raise ValueError(f"Expected 50 unique tasks for {suite}, got {len(task_ids)} / {len(set(task_ids))}")
        for task_id in task_ids:
            key = (suite, task_id)
            if key in selected_keys:
                raise ValueError(f"Duplicate task key: {key}")
            selected_keys.add(key)

        manifest = {
            "schema_version": 1,
            "benchmark_mode": "plus",
            "suite": suite,
            "categories": list(CATEGORIES),
            "difficulties": list(DIFFICULTIES),
            "seed": args.seed,
            "sampling": "suite_category_difficulty_stratified_sha256_rank",
            "quotas": quotas,
            "candidate_counts": candidate_counts,
            "task_ids": task_ids,
        }
        (args.output_root / f"{suite}.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )

    if total_by_difficulty != {4: 100, 5: 100} or len(selected_keys) != 200:
        raise ValueError(f"Invalid totals: {total_by_difficulty=}, unique={len(selected_keys)}")
    print(json.dumps({"total": len(selected_keys), "by_difficulty": total_by_difficulty}, sort_keys=True))


if __name__ == "__main__":
    main()
