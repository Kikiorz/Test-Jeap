"""Turn LIBERO-Plus episode journals into a compact markdown report.

The heavy lifting stays in ``examples/libero/main.py`` (the same summariser the
evaluation sweep ends with); this module only shells out to it and formats the
printed lines, so the numbers reported here always match the official summary.

Usage:
    python scripts/libero_plus_report.py --run-id pi05-plus-30k
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULT_RE = re.compile(
    r"^(?P<label>.+?): successes=(?P<successes>\d+) episodes=(?P<episodes>\d+) "
    # Per-suite lines omit the failure count; category and difficulty lines carry it.
    r"(?:failures=(?P<failures>\d+) )?errors=(?P<errors>\d+) pending=(?P<pending>\d+) "
    r"success_rate=(?P<rate>[0-9.]+)(?: official=(?P<official>\w+))?$"
)


def summarise(run_id: str) -> str:
    log_root = ROOT / "data" / "libero-eval" / run_id
    journals = sorted(log_root.glob("plus-*.shard-*-of-*.jsonl"))
    if not journals:
        raise SystemExit(f"no journals found under {log_root}")
    cmd = ["bash", str(ROOT / "scripts" / "run_libero_evaluation.sh"), "summary", *map(str, journals)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"summary failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def parse(output: str):
    suites: dict[str, float] = {}
    categories: dict[str, float] = {}
    difficulties: dict[str, float] = {}
    aggregate: dict[str, str] = {}
    section = None
    for raw in output.splitlines():
        line = raw.split("INFO:root:")[-1].strip()
        if line.startswith("LIBERO-Plus category summary"):
            section = "category"
            continue
        if line.startswith("LIBERO-Plus difficulty summary"):
            section = "difficulty"
            continue
        if line.startswith("Aggregate:"):
            aggregate = {k: v for k, v in re.findall(r"(\w+)=([0-9.]+)", line)}
            continue
        match = RESULT_RE.match(line)
        if not match:
            continue
        rate = float(match.group("rate"))
        if match.group("label") in {"Aggregate", "Selected"}:
            continue
        if section == "category":
            categories[match.group("label")] = rate
        elif section == "difficulty":
            difficulties[match.group("label")] = rate
        else:
            suites[match.group("label")] = rate
    return suites, categories, difficulties, aggregate


def markdown(run_id: str, suites, categories, difficulties, aggregate) -> str:
    lines = [f"# LIBERO-Plus full sweep — `{run_id}`", ""]
    lines += ["| Suite | Success |", "|:--|--:|"]
    lines += [f"| {name} | {100 * rate:.1f} |" for name, rate in suites.items()]
    overall = aggregate.get("success_rate")
    if overall is not None:
        lines.append(f"| **All suites (task-micro)** | **{100 * float(overall):.1f}** |")
    lines += ["", "| Perturbation category | Success |", "|:--|--:|"]
    lines += [f"| {name} | {100 * rate:.1f} |" for name, rate in categories.items()]
    if categories:
        macro = sum(categories.values()) / len(categories)
        lines.append(f"| **Category-macro** | **{100 * macro:.1f}** |")
    lines += ["", "| Difficulty | Success |", "|:--|--:|"]
    lines += [f"| L{name} | {100 * rate:.1f} |" for name, rate in sorted(difficulties.items())]
    if aggregate:
        details = " ".join(f"{k}={v}" for k, v in aggregate.items())
        lines += ["", f"`{details}`"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = summarise(args.run_id)
    report = markdown(args.run_id, *parse(output))
    if args.output:
        pathlib.Path(args.output).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
