"""Dependency-free live Con1 dashboard and lossless numeric journals."""

import csv
import html
import json
import math
import os
from pathlib import Path

GROUPS = {
    "ΔZ 预测头": (
        "NMSE<1 表示优于零预测；余弦、幅度和时间变化需一起看。",
        ("rapr_delta_nmse", "rapr_delta_cosine", "rapr_prediction_target_norm_ratio",
         "rapr_prediction_temporal_spread", "rapr_target_temporal_spread", "rapr_head_grad_norm")),
    "动作条件检索": (
        "gain = 对照误差 − 正常误差。后两层版本中，这里只消融最后一次检索，保留第一次；不是两层同时消融。KL 大不代表更有效。",
        ("rapr_uniform_flow7_gain", "rapr_unconditioned_flow7_gain", "rapr_action_condition_kl",
         "rapr_retrieval_max_mass", "rapr_router_grad_norm")),
    "后两层检索（仅 late-two 实验）": (
        "17/18 是正式18层专家中的位置。第一层 gain 比较仅首次残差与无残差；第二层 gain 比较两次残差与仅首次残差。是同一批次的顺序增益，不是闭环成功率。",
        ("rapr_layer17_residual_rms", "rapr_layer18_residual_rms", "rapr_layer17_velocity_effect_rms",
         "rapr_layer17_flow7_gain", "rapr_layer18_flow7_gain", "rapr_layer_routes_l1",
         "rapr_layer17_route_entropy", "rapr_layer18_route_entropy")),
    "控制相关性监督 r × s": (
        "有效 horizon 接近 H 是近似均匀；变小代表集中，不自动等于进步。梯度比为批次平均 RMS 之比，仅比较 LA 与加权 LΔ。",
        ("rapr_weight_effective_horizon", "rapr_q_uniform_l1", "rapr_action_sensitivity",
         "rapr_delta_gradient_ratio_aggregate", "rapr_sensitivity_nonzero_fraction")),
    "连续残差 α": (
        "α=0.05 不代表动作改变了 5%。看实际修正量与局部 flow gain；gain 为负表示此批数据残差有害。",
        ("rapr_route_gate", "rapr_raw_velocity_correction_rms", "rapr_residual_flow7_gain",
         "rapr_nonregression_loss", "rapr_alpha_update_norm")),
    "动作专家与原模型参照": (
        "第一阶段动作梯度/更新应为零。第二阶段参考始终是原始 JEPA-WAM，不是当前专家的 α=0 分支。",
        ("flow_mse_action7", "reference_flow_mse_action7", "reference_velocity_action7_rms",
         "rapr_action_grad_norm", "rapr_action_relative_update")),
    "四类任务": (
        "训练批次误差受样本难度影响；跨 checkpoint 以固定留出集和同种子闭环 rollout 为准。",
        ("libero_10_flow_loss", "libero_goal_flow_loss", "libero_object_flow_loss", "libero_spatial_flow_loss")),
}

MINIMAL_GROUPS = {
    "训练状态（非额外模块诊断）": (
        "保留损失组成、学习率、α和梯度有限性检查；全量权重分析在每1k的CPU审计中进行。",
        ("loss", "flow_loss", "weighted_vjepa_loss", "weighted_rapr_prediction_loss",
         "weighted_rapr_nonregression_loss", "con1_learning_rate", "action_learning_rate",
         "alpha_learning_rate", "rapr_route_gate", "grad_norm")),
    "三个额外模块指标": (
        "NMSE<1表示优于零预测；q偏离均匀表示监督权重有区分，不自动等于更好；残差gain为正表示本批7维动作误差下降。这些都不是闭环成功率。",
        ("rapr_delta_nmse", "rapr_q_uniform_l1", "rapr_residual_flow7_gain")),
}


def derived_metrics(record):
    """Derive a ratio of mean RMS values without averaging undefined ratios.

    Episode-tail samples can have zero prediction gradient and nonzero action
    gradient. The legacy mean of per-sample ratios is therefore not useful.
    Keep that raw metric untouched for audit; give the new statistic a new name.
    """
    result = dict(record)
    numerator = result.get("rapr_delta_action_gradient_rms")
    denominator = result.get("rapr_delta_prediction_gradient_rms")
    if numerator is not None and denominator is not None:
        ratio = numerator / denominator if denominator > 0 else float("nan")
        valid = math.isfinite(ratio)
        result["rapr_delta_gradient_ratio_aggregate"] = ratio if valid else 0.0
        result["rapr_delta_gradient_ratio_aggregate_valid"] = float(valid)
    return result


def visible_value(row, key):
    if key == "rapr_delta_gradient_ratio_aggregate" and not row.get(key + "_valid", 0):
        return None
    if key.startswith("libero_") and row.get(key.removesuffix("_flow_loss") + "_sample_fraction", 0) == 0:
        return None
    return row.get(key)


def atomic_text(path, text):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def chart(rows, key):
    points = [(row["global_completed_updates"], value) for row in rows
              if (value := visible_value(row, key)) is not None]
    if not points:
        return "<p>尚无数据</p>"
    # Smooth the full sequence before downsampling for display. Raw values are
    # always retained in JSONL/CSV and the gray curve.
    ema = points[0][1]
    smooth = []
    for x, y in points:
        ema = .9 * ema + .1 * y
        smooth.append((x, ema))
    keep = list(range(0, len(points), max(1, len(points) // 220)))
    if keep[-1] != len(points) - 1:
        keep.append(len(points) - 1)
    raw, avg = [points[i] for i in keep], [smooth[i] for i in keep]
    xmin, xmax = points[0][0], points[-1][0]
    ymin = min(y for _, y in raw + avg)
    ymax = max(y for _, y in raw + avg)
    if key.endswith("gain"):
        ymin, ymax = min(0, ymin), max(0, ymax)
    pad = max((ymax-ymin)*.05, max(abs(ymax), 1e-15)*.01)
    ymin, ymax = ymin-pad, ymax+pad
    def xy(point):
        x, y = point
        return f"{58+330*(x-xmin)/max(xmax-xmin,1):.2f},{145-120*(y-ymin)/(ymax-ymin):.2f}"
    return (f'<svg viewBox="0 0 410 180" role="img" aria-label="{html.escape(key)}">'
            f'<text x="2" y="26">{ymax:.2g}</text><text x="2" y="145">{ymin:.2g}</text>'
            '<path d="M58 20V145H390" fill="none" stroke="#cbd5e1"/>'
            f'<polyline points="{" ".join(map(xy, raw))}" fill="none" stroke="#cbd5e1" stroke-width="1"/>'
            f'<polyline points="{" ".join(map(xy, avg))}" fill="none" stroke="#0f766e" stroke-width="2"/>'
            f'<text x="58" y="170">{xmin}</text><text x="345" y="170">{xmax}</text></svg>')


class PaperMetricsWriter:
    def __init__(self, directory, *, completed_updates=0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.journal = self.directory / "training_metrics.jsonl"
        self.history = {}
        if self.journal.exists():
            contents = self.journal.read_text()
            lines = contents.splitlines()
            # A live reader can observe an append before its final newline.
            if contents and not contents.endswith("\n"):
                lines = lines[:-1]
            for line in lines:
                row = json.loads(line)
                if row["completed_updates"] <= completed_updates:
                    self.history[row["completed_updates"]] = derived_metrics(row)

    def append(self, record):
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in record.values()):
            raise ValueError("All training metrics must be finite numeric values")
        record = derived_metrics(record)
        with self.journal.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        csv_path = self.directory / "training_metrics.csv"
        new_file = not csv_path.exists()
        with csv_path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(("global_completed_updates", "unix_time", "metric", "value"))
            writer.writerows((record["global_completed_updates"], record["unix_time"], key, value)
                             for key, value in record.items())
        self.history[record["completed_updates"]] = record
        self.render()

    def document(self):
        """Render a read-only live view; never rewrite the training journal."""
        rows = [self.history[key] for key in sorted(self.history)]
        if not rows:
            return "<!doctype html><meta charset='utf-8'><p>尚无已提交的训练指标。</p>"
        latest = rows[-1]
        nonregression_disabled = latest.get("rapr_nonregression_weight") == 0
        sections = []
        minimal = "rapr_delta_nmse" in latest and "rapr_delta_cosine" not in latest
        groups = MINIMAL_GROUPS if minimal else GROUPS
        for name, (description, keys) in groups.items():
            figures = []
            for key in keys:
                value = visible_value(latest, key)
                shown = "暂无" if value is None else f"{value:.6g}"
                if nonregression_disabled and key in ("rapr_nonregression_loss", "weighted_rapr_nonregression_loss"):
                    shown = "已关闭（未计算）"
                figures.append(f'<article><h3>{html.escape(key)}</h3><strong>{shown}</strong>{chart(rows,key)}</article>')
            sections.append(f'<section><h2>{html.escape(name)}</h2><p>{html.escape(description)}</p>'
                            f'<div class="grid">{"".join(figures)}</div></section>')
        document = ('<!doctype html><html lang="zh"><meta charset="utf-8"><meta http-equiv="refresh" content="30">'
                    '<title>Con1 训练指标</title><style>body{font:15px system-ui;max-width:1500px;margin:32px auto;padding:0 24px;color:#172b3a;background:#f8fafc}'
                    '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:16px}'
                    'article{background:white;border:1px solid #dbe3eb;border-radius:8px;padding:14px}h3{font-size:12px;overflow-wrap:anywhere}'
                    'svg{width:100%;font:10px system-ui}strong{font-size:21px}section{margin-top:30px}</style>'
                    f'<h1>Con1 模块训练诊断 · global step {latest["global_completed_updates"]}</h1>'
                    '<p>这是训练诊断，不是闭环成功率。灰线：原始批次值；绿线：EMA(0.9)。每 30 秒刷新页面。</p>'
                    + ('<p>当前已关闭每步原始基线前向及 non-regression loss；缺失值不是零误差。原始基线仍用于定点评测。</p>'
                       if nonregression_disabled else '')
                    +
                    '<p><a href="training_metrics.csv">下载全部指标 CSV（长表）</a> · <a href="training_metrics.jsonl">原始 JSONL</a></p>'
                    + "".join(sections) + '</html>')
        return document

    def render(self):
        if not self.history:
            return
        latest = self.history[max(self.history)]
        atomic_text(self.directory / "training_dashboard.html", self.document())
        atomic_text(self.directory / "latest_metrics.json", json.dumps(latest, indent=2) + "\n")
