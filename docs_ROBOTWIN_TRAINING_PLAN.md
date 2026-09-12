# RoboTwin 2.0 · Con1/Con2 训练方案

在与 JEPA-WAM 论文相同的 **20 个 RoboTwin 任务**上，从官方 JEPA-WAM π0.5
`19999` checkpoint 出发，做 Con1/Con2 的两臂消融。数据管线与动作约定的由来见
`docs_ROBOTWIN_CON1_SETUP.md`。

## 1. 已确定的设定

| 项 | 值 |
|---|---|
| 基座 | `jepa_wam_pi05_robotwin_publish/pi05_robotwin_clean_20_vjepa_aux/19999`（发布态转换后） |
| 数据 | 官方 RoboTwin 2.0 LeRobot v3 → v2.1：2,500 episodes / 548,985 帧 / 3 相机 / 14 维 |
| 动作空间 | **逐帧增量** `a[t] − s[t]`（由基座自身的 flow loss 判定：0.0224 vs 其他候选 ≥0.19；应用后整库 loss 0.3926 → 0.0739） |
| proprioception | 基座 `discrete_state_input=False`，**不消费 state**（state 三种变体 loss 完全相同） |
| 训练集 | **不做 holdout**（`con1_holdout_fraction=0.0`）：全部 2,500 episodes 参与训练；官方基准数据即测试集 |
| 相机 | `cam_high` / `cam_left_wrist` / `cam_right_wrist`（policy 键 `observation/image`、`wrist_image`、`wrist_image_right`） |
| 模型形状 | `vjepa_num_queries=64`、`vjepa_target_dim=1408`、`con1_latent_dim=4224`（3 视角 × 1408）、`action_horizon=50`、`con1_action_dims=14` |

## 2. 两臂设计

两臂**只有新增模块不同**，其余（数据、batch、lr、seed、步数、阶段）完全一致：

| 臂 | 配置名 | Con1 livecross | Con2 精修 | 全前缀 VLM 上下文 |
|---|---|---|---|---|
| A | `pi05_robotwin_con1_livecross_20k` | ✅ | ❌ | ❌ |
| B | `pi05_robotwin_con1con2_ctx_20k` | ✅ | ✅ | ✅ |

两者共用：`con1_action_adapter=True`、`con1_cross_attention_out_init=0.02`、
`con1_alpha_initial=0.3`、`con1_residual_budget=0.05`、`con1_delta_weight=0.2`、
flow 权重 2.0→1.0、`con1_stage1_steps=2000`（之后动作专家 14–17 层解锁）。

## 3. 训练配方

```
batch 64, lr 1e-5 (cosine), con1_lr_multiplier 2,
num_workers 12（RoboTwin 每帧解 AV1，必须多进程读数据，否则卡死）,
save_interval 1000, 4×GPU
```

启动器：`scripts/run_robotwin_arms.sh`（可用 `STEPS=`、`NUM_WORKERS=` 覆盖）。

**步数待定**：LIBERO（`action_horizon=10`）在 4 卡上是 ~1.7 s/step；RoboTwin 的
`action_horizon=50` 会让动作专家的序列长 5 倍，预计每步数倍于此。因此先用
**20 步冒烟测试**实测步时，再据此确定"每臂多少步"，而不是直接照搬 LIBERO 的 4500 步。

## 4. 评测

1. **机制指标（不需要仿真器，缓存好即可做）**：
   `scripts/probe_con1_budget_and_conditioning.py` 的配对 flow 探针——同一 checkpoint
   在 `--zero-correction`（精确基座）与 Con1 打开之间比 held-out flow，报告相对增益与 σ。
2. **闭环（需要先装 RoboTwin 仿真器）**：20 任务 × Clean/Random，与论文口径一致；
   基线可用同一基座 checkpoint 直接跑，配对比较。

## 5. 当前进度

* [x] 代码分支 `feat/RoboTwin`（35 项 Con1 单测通过）
* [x] 基座 checkpoint 发布态转换 + 形状对齐
* [x] 数据 v3 → v2.1 转换（含 7,500 个 per-episode 视频）
* [x] 动作约定判定与落地（逐帧增量）+ 基座 loss 复验
* [x] 两臂配置与启动器
* [ ] Con1 缓存（stage 1 帧状态 4 卡 8 worker 运行中 → stage 2 R tokens）
* [ ] 20 步冒烟测试（实测步时）→ 确认步数预算
* [ ] 两臂训练（4500 步或按实测调整）
* [ ] 配对 flow 探针出机制指标
* [ ] （可选）装 RoboTwin 仿真器做闭环

## 6. 已知风险

* **AV1 解码是数据瓶颈**：缓存阶段 GPU 利用率仅 0–38%，CPU 打满；16 worker 会因线程
  超订直接卡死（load 553），8 worker 是当前的最优点（~14 episodes/分钟）。
* **训练期数据加载**：LeRobot 不缓存解码帧，因此必须 `num_workers>0`；否则训练会被
  视频解码拖死。
* **步时**：horizon 50 的 step 成本明显高于 LIBERO，需实测后再定步数。

## 7. 两臂结果（2026-09-12）

配置最终落地：`action_horizon=16`（权重里不含 horizon，16 比 50 快约 3 倍）、
`con1_holdout_fraction=0.0`（全 2500 集训练）、批大小 64、4500 步/臂、
数据为 384² 内联图像副本（训练不再解码 AV1）。

两条重要的工程结论：

* **训练速度**：内联图像让 batch 64 的步时从「视频解码下的 ~6–7 s/step(batch 8)」
  变成 **~0.7 s/step**，4500 步/臂约 50 分钟；
* **缓存构建**：teacher 状态改用同一批视频抽出的 384² JPEG 重建后，
  速率从 8 集/分钟提升到 **29 集/分钟**（2500 集约 1.5 小时）。

### 配对 flow 探针（48 组配对批次，同一数据加载器与种子）

`base` = A 臂 checkpoint 把 Con1 修正置零（`--zero-correction`），因此差值**只反映
Con1 路径本身**，不含微调动作专家带来的收益。

| 配置 | flow | correction RMS | 相对 base | 显著性 |
|---|---|---|---|---|
| base（Con1 修正置零） | 0.004157 | 0.028 | — | — |
| **A 臂：Con1 livecross** | **0.003667** | 0.524 | **−11.77%** | **t = 7.0** |
| **B 臂：Con1 + Con2 + 全前缀上下文** | **0.003664** | 0.524 | **−11.85%** | **t = 7.9** |
| B vs A（配对） | — | — | −0.10% | t = 0.16（不显著） |

结论：

1. **Con1 在 RoboTwin 上的机制收益远大于 LIBERO**（−11.8% vs −1.9%），7σ 级别；
2. **Con2 精修模块与全前缀 VLM 上下文在这一设置下不带来额外收益**（+0.1%，t=0.16）——
   两者与 Con1 livecross 等价，所以推荐部署更简单的 A 臂配置；
3. 注意这是**机制指标**（训练分布批次的 flow loss），不是闭环成功率；闭环需要
   RoboTwin 仿真器（尚未安装）。
