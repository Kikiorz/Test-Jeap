"""Immutable evaluation panels for four-suite paper Con1 training."""

from collections import defaultdict
import hashlib
import json
import random

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
CATEGORIES = ("Background Textures", "Robot Initial States", "Camera Viewpoints",
              "Language Instructions", "Sensor Noise", "Objects Layout", "Light Conditions")
MILESTONES = {1: (1001, 5000), 2: (1001, 5001, 10001, 15000)}
# Completed updates, not zero-based checkpoint names. Global: 5k, 10k, 15k, 20k.
DIRECT_DELTA_MILESTONES = {1: (5000,), 2: (5000, 10000, 15000)}


def choose_panels(classification, *, seed=731, monitor_per_category=3, final_per_category=10):
    """Category- and difficulty-stratified, disjoint monitor/final task IDs."""
    if min(monitor_per_category, final_per_category) < 1:
        raise ValueError("Panel sizes must be positive")
    panels = {"monitor": {}, "final": {}}
    for suite in SUITES:
        rows = classification[suite]
        if [row["id"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"Classification IDs for {suite} are not contiguous one-based IDs")
        by_category = defaultdict(list)
        for row in rows:
            by_category[row["category"]].append(row)
        if set(by_category) != set(CATEGORIES):
            raise ValueError(f"Unexpected interference categories for {suite}")
        monitor, final = [], []
        for category in CATEGORIES:
            salt = int.from_bytes(hashlib.sha256(f"{seed}:{suite}:{category}".encode()).digest()[:8], "big")
            rng = random.Random(salt)
            levels = defaultdict(list)
            for row in by_category[category]:
                # Language/layout variants may have no numeric difficulty.
                # Treat these as their own stratum; preserve None in metadata.
                level = -1 if row["difficulty_level"] is None else int(row["difficulty_level"])
                levels[level].append(row)
            for bucket in levels.values():
                rng.shuffle(bucket)
            level_order = sorted(levels)
            rng.shuffle(level_order)
            ordered = []
            while any(levels.values()):
                for level in level_order:
                    if levels[level]:
                        ordered.append(levels[level].pop())
            if len(ordered) < monitor_per_category + final_per_category:
                raise ValueError(f"Too few tasks in {suite}/{category}")
            monitor.extend(ordered[:monitor_per_category])
            final.extend(ordered[monitor_per_category:monitor_per_category + final_per_category])
        convert = lambda row: {"task_id": int(row["id"]) - 1, "name": row["name"],
                               "category": row["category"], "difficulty_level": row["difficulty_level"]}
        panels["monitor"][suite] = sorted(map(convert, monitor), key=lambda row: row["task_id"])
        panels["final"][suite] = sorted(map(convert, final), key=lambda row: row["task_id"])
        if {row["task_id"] for row in panels["monitor"][suite]} & {row["task_id"] for row in panels["final"][suite]}:
            raise AssertionError("Monitor and final panels overlap")
    return {"schema_version": 1, "selection_seed": seed,
            "monitor_episode_seed": 431, "final_episode_seed": 743,
            "monitor_standard_trials_per_task": 2,
            "note": "Monitor panel selects checkpoints; disjoint final Plus panel is evaluated only at the end.",
            "classification_sha256": hashlib.sha256(json.dumps(classification, sort_keys=True).encode()).hexdigest(),
            **panels}


def summarize_journals(paths):
    totals = defaultdict(lambda: {"success": 0, "failure": 0, "error": 0, "steps_sum": 0})
    latest = {}
    for path in paths:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("record_type") == "episode":
                    latest[(path.stem, row["task_suite_name"], row["task_id"], row["episode_idx"])] = row
    for row in latest.values():
        key = f"{row['task_suite_name']}/{row.get('category') or 'standard'}"
        group = totals[key]
        status = row["status"]
        if status not in ("success", "failure", "error"):
            raise ValueError(f"Unknown outcome status {status}")
        group[status] += 1
        group["steps_sum"] += int(row.get("num_steps", 0))
    for group in totals.values():
        count = sum(group[key] for key in ("success", "failure", "error"))
        group["count"] = count
        group["success_rate"] = group["success"] / count if count else None
        group["mean_steps"] = group["steps_sum"] / count if count else None
    return dict(totals)


def paired_results(baseline_paths, candidate_paths):
    """Pair identical environment/policy seeds; errors are never successes."""
    def read(paths):
        result, contracts = {}, {}
        for path in paths:
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get("record_type") == "run":
                    contract = dict(row["run_config"])
                    contract.pop("run_id", None)
                    contracts[path.stem] = contract
                elif row.get("record_type") == "episode":
                    result[(path.stem, row["task_suite_name"], row["task_id"], row["episode_idx"])] = row
        return result, contracts
    baseline, base_contracts = read(baseline_paths)
    candidate, candidate_contracts = read(candidate_paths)
    if base_contracts != candidate_contracts:
        raise ValueError("Cannot pair evaluations with different environment/seed contracts")
    if baseline.keys() != candidate.keys():
        raise ValueError("Paired evaluations have different episode keys")
    groups = defaultdict(lambda: {"both_success": 0, "gain": 0, "regression": 0, "both_failure": 0, "error": 0})
    for key, base in baseline.items():
        other = candidate[key]
        for field in ("episode_seed", "task_name", "category", "difficulty_level", "max_steps"):
            if base.get(field) != other.get(field):
                raise ValueError(f"Paired episode identity differs: {key}: {field}")
        if "error" in (base["status"], other["status"]):
            outcome = "error"
        elif base["status"] == other["status"]:
            outcome = "both_success" if base["status"] == "success" else "both_failure"
        else:
            outcome = "gain" if other["status"] == "success" else "regression"
        mode = key[0].split("_", 1)[0]
        for group in ("all", mode, f"{mode}/{base['task_suite_name']}",
                      f"{mode}/{base['task_suite_name']}/{base.get('category') or 'standard'}"):
            groups[group][outcome] += 1
    for group in groups.values():
        group["count"] = sum(group.values())
        n = group["count"]
        group["baseline_success_rate"] = (group["both_success"] + group["regression"]) / n if n else None
        group["candidate_success_rate"] = (group["both_success"] + group["gain"]) / n if n else None
        group["success_rate_delta"] = (group["gain"] - group["regression"]) / n if n else None
    return {"complete": bool(baseline) and groups["all"]["error"] == 0, "groups": dict(groups)}
