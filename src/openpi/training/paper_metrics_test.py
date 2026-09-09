import json

import pytest

from openpi.training.paper_metrics import PaperMetricsWriter, derived_metrics, visible_value


def test_dashboard_logs_raw_values_and_resumes_without_stale_curve(tmp_path):
    def record(step):
        return {"step": step-1, "completed_updates": step, "global_completed_updates": step,
                "unix_time": 1234., "rapr_uniform_flow7_gain": -1e-8, "rapr_route_gate": .05}
    writer = PaperMetricsWriter(tmp_path)
    writer.append(record(1))
    writer.append(record(11))
    resumed = PaperMetricsWriter(tmp_path, completed_updates=1)
    assert set(resumed.history) == {1}
    resumed.append(record(2))
    assert json.loads((tmp_path / "latest_metrics.json").read_text())["completed_updates"] == 2
    assert "global step 2" in (tmp_path / "training_dashboard.html").read_text()
    assert "<svg" in (tmp_path / "training_dashboard.html").read_text()
    assert "-1e-08" in (tmp_path / "training_metrics.csv").read_text()
    with pytest.raises(ValueError, match="finite"):
        resumed.append(dict(record(3), loss=float("nan")))


def test_gradient_ratio_uses_aggregate_rms_and_preserves_raw():
    raw = {"rapr_delta_action_gradient_rms": 7.3e-13,
           "rapr_delta_prediction_gradient_rms": 3.9e-7,
           "rapr_delta_action_prediction_gradient_ratio": 2e15}
    corrected = derived_metrics(raw)
    assert corrected["rapr_delta_gradient_ratio_aggregate"] == pytest.approx(7.3e-13 / 3.9e-7)
    assert corrected["rapr_delta_action_prediction_gradient_ratio"] == 2e15
    assert "rapr_delta_gradient_ratio_aggregate" not in raw
    missing = derived_metrics(dict(raw, rapr_delta_prediction_gradient_rms=0.0))
    assert visible_value(missing, "rapr_delta_gradient_ratio_aggregate") is None


def test_live_view_ignores_unfinished_append_without_rewriting_raw(tmp_path):
    journal = tmp_path / "training_metrics.jsonl"
    record = {"completed_updates": 1, "global_completed_updates": 1, "unix_time": 1.,
              "rapr_delta_action_gradient_rms": 1e-12, "rapr_delta_prediction_gradient_rms": 1e-6}
    contents = json.dumps(record) + '\n{"completed_updates":'
    journal.write_text(contents)
    view = PaperMetricsWriter(tmp_path, completed_updates=1)
    assert view.history[1]["rapr_delta_gradient_ratio_aggregate"] == pytest.approx(1e-6)
    assert "rapr_delta_gradient_ratio_aggregate" in view.document()
    assert journal.read_text() == contents


def test_minimal_dashboard_shows_only_three_extra_diagnostics(tmp_path):
    writer = PaperMetricsWriter(tmp_path)
    writer.append({"completed_updates": 1001, "global_completed_updates": 1001, "unix_time": 1.,
                   "rapr_delta_nmse": .9, "rapr_delta_cosine": .2, "rapr_route_gate": .05})
    writer.append({"completed_updates": 1011, "global_completed_updates": 1011, "unix_time": 2.,
                   "loss": .03, "grad_norm": .01, "rapr_route_gate": .05,
                   "rapr_delta_nmse": .89, "rapr_q_uniform_l1": .1, "rapr_residual_flow7_gain": 1e-6})
    document = writer.document()
    assert "三个额外模块指标" in document
    for key in ("rapr_delta_nmse", "rapr_q_uniform_l1", "rapr_residual_flow7_gain"):
        assert f"<h3>{key}</h3>" in document
    for removed in ("rapr_delta_cosine", "rapr_uniform_flow7_gain", "rapr_head_grad_norm", "rapr_layer17_residual_rms"):
        assert f"<h3>{removed}</h3>" not in document
    assert "rapr_delta_cosine" in (tmp_path / "training_metrics.jsonl").read_text()


def test_disabled_nonregression_is_not_reported_as_zero_error(tmp_path):
    writer = PaperMetricsWriter(tmp_path)
    writer.append({"completed_updates": 3001, "global_completed_updates": 8001, "unix_time": 1.,
                   "rapr_delta_nmse": .84, "weighted_rapr_nonregression_loss": .0001})
    writer.append({"completed_updates": 3011, "global_completed_updates": 8011, "unix_time": 2.,
                   "rapr_delta_nmse": .83, "rapr_nonregression_weight": 0.,
                   "rapr_q_uniform_l1": .2, "rapr_residual_flow7_gain": 1e-6})
    assert "已关闭（未计算）" in writer.document()
    assert "缺失值不是零误差" in writer.document()
    latest = json.loads((tmp_path / "latest_metrics.json").read_text())
    assert "weighted_rapr_nonregression_loss" not in latest
    assert "rapr_nonregression_loss" not in latest
