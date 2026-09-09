from collections import Counter
import json

import pytest

from paper_con1_protocol import CATEGORIES, DIRECT_DELTA_MILESTONES, SUITES, choose_panels, paired_results


def test_direct_restart_evaluates_only_exact_global_5k_milestones():
    global_steps = [step + (5000 if stage == 2 else 0)
                    for stage in (1, 2) for step in DIRECT_DELTA_MILESTONES[stage]]
    assert global_steps == [5000, 10000, 15000, 20000]
    assert [step - 1 for step in DIRECT_DELTA_MILESTONES[2]] == [4999, 9999, 14999]


def test_panels_cover_all_categories_and_do_not_overlap():
    classification = {}
    for suite in SUITES:
        rows = []
        for category in CATEGORIES:
            for i in range(30):
                rows.append({"id": len(rows) + 1, "name": f"task_{len(rows)}", "category": category,
                             "difficulty_level": None if i % 6 == 0 else 1 + i % 5})
        classification[suite] = rows
    panels = choose_panels(classification)
    assert panels == choose_panels(classification)
    for suite in SUITES:
        monitor, final = panels["monitor"][suite], panels["final"][suite]
        assert len(monitor) == 21 and len(final) == 70
        assert not {row["task_id"] for row in monitor} & {row["task_id"] for row in final}
        assert Counter(row["category"] for row in monitor) == {category: 3 for category in CATEGORIES}
        assert Counter(row["category"] for row in final) == {category: 10 for category in CATEGORIES}


def test_pairs_separate_gains_regressions_and_reject_seed_changes(tmp_path):
    paths = []
    for variant, statuses in (("base", ["success", "failure", "success", "failure"]),
                              ("candidate", ["failure", "success", "success", "failure"])):
        folder = tmp_path / variant
        folder.mkdir()
        path = folder / "plus_libero_goal.jsonl"
        rows = [{"record_type": "run", "run_config": {"run_id": variant, "seed": 431}}]
        rows += [{"record_type": "episode", "task_suite_name": "libero_goal", "task_id": i,
                  "episode_idx": 0, "episode_seed": 123 + i, "status": status, "category": "Sensor Noise"}
                 for i, status in enumerate(statuses)]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        paths.append(path)
    result = paired_results(paths[:1], paths[1:])
    assert result["complete"]
    assert result["groups"]["all"]["gain"] == result["groups"]["all"]["regression"] == 1
    assert result["groups"]["all"]["success_rate_delta"] == 0
    paths[1].write_text(paths[1].read_text().replace('"episode_seed": 123', '"episode_seed": 999'))
    with pytest.raises(ValueError, match="identity differs"):
        paired_results(paths[:1], paths[1:])
