# RoboTwin 2.0：数据管线与动作约定（Con1/Con2 前置）

这份文档记录把官方 RoboTwin 2.0 数据接到本项目训练管线时踩到的每一个坑，以及
**动作约定是怎么被判定的**——这一步不做对，后面所有训练都是在错误的动作空间里进行。

## 1. 基座 checkpoint 的布局

官方发布的是**训练态** checkpoint（`<step>/_CHECKPOINT_METADATA` 描述
`assets`/`params`/`train_state` 三个条目，`params/` 里没有 `_CHECKPOINT_METADATA`）。
openpi 的推理与权重加载路径走的是**发布态**布局（`restore_params(<dir>/params)`，
读取 `metadata["params"]`），直接用会报错。

处理：

* `scripts/publish_robotwin_checkpoint.py` —— 用 tensorstore 直接读 ocdbt/zarr 参数树
  （键名就是树路径，58 个叶子），再用 `PyTreeCheckpointer` 写成 `{"params": tree}`；
* `src/openpi/models/model.py` 的 `restore_params` 增加兼容分支：新版 orbax 对裸
  pyTree checkpoint 返回 `StepMetadata`，此时按 `item=None` 还原并解包 `["params"]`。

**形状核对**（与 `docs_ROBOTWIN_CON1_TRAINING.md` 中"16 queries"的说法不符，以权重为准）：

| 参数 | 形状 |
|---|---|
| `vjepa_query_tokens` | **(64, 2048)** |
| `vjepa_alignment_out` | (2048, **1408**) |
| `action_in_proj` / `action_out_proj` | (32, 1024) / (1024, 32) |

因此配置用 `vjepa_num_queries=64, vjepa_query_grid_size=8, vjepa_target_dim=1408`。

## 2. 数据集：v3 → v2.1

openpi 钉住的 lerobot 是 `CODEBASE_VERSION = v2.1`（`meta/tasks.jsonl`、逐个 episode 的
parquet、每个 episode 一个 mp4），而官方发布是 v3.0（`meta/tasks.parquet`、按 chunk 打包的
parquet 与 mp4）。`scripts/convert_robotwin_v3_to_v21.py` 做转换，`scripts/split_robotwin_videos.py`
按 `meta/episodes` 里的时间戳用 `ffmpeg -c copy` 把视频切成 2500×3 个 per-episode mp4。

转换中必须处理的细节：

1. **时间戳是全局的**：v3 的 `from_timestamp` 是数据集级时间，而 mp4 是按文件切的，
   必须归一到该文件内；切完 per-episode 视频后统一写成 `i / fps`。
2. **视频按 episode chunk 索引**：v2.1 模板是 `videos/chunk-{episode_chunk:03d}/...`，
   即 `episode_index // chunks_size`，不是固定的 chunk-000。
3. **列名**：openpi 用 `actions`（与 LIBERO 数据集一致），官方是 `action`；
   同时 `delta_timestamps` 在数据集层就按 `action_sequence_keys=["actions"]` 取块。
4. **统计结构**：`meta/episodes_stats.jsonl` 每条是
   `{"episode_index": e, "stats": {feature: {stat: value}}}`，且每个**叶子**统计量至少一维。
5. **本地数据集**：`repo_id` 用本地路径时，要把路径作为 `root` 传给 LeRobot，
   并跳过它对 Hub 的版本查询与 `get_safe_version`；时间戳容差放宽到 0.1s。

## 3. 动作约定判定（关键）

官方数据的 `action[t] == state[t+1]`（绝对关节目标），而基座动作统计 mean≈0、std 0.11–1.0，
说明两者的动作空间不同。判定方法是**问模型自己**：固定同一批帧、prompt、flow 时刻与噪声，
只换动作约定，看基座的 flow-matching loss（`scripts/probe_robotwin_convention.py`）。

候选（`states` 为 t..t+H-1 的关节位置，`actions` 为对应的绝对下一帧目标）：

| 约定 | 定义 | flow loss |
|---|---|---|
| abs | a[t+i] | 0.2455 |
| delta_now | a[t+i] − s[t] | 0.1949 |
| **delta_step** | **a[t+i] − s[t+i]** | **0.0224** |
| delta_end | a[t+i] − s[t+H−1] | 0.2392 |
| delta_mean | a[t+i] − mean(s) | 0.3634 |
| scale2 / window2 | (a−s)×2 / s[t+i+2]−s[t+i] | 0.0351 / 0.0350 |
| scale5 / window8 | ×5 / 8 帧窗口 | 0.1123 / 0.1853 |

结论：**基座的动作空间是逐帧关节增量 `a[t] − s[t]`（即关节速度），原速、不缩放**；
放大或加长窗口都更差。0.0224 与 LIBERO 基座的同口径 0.0172 同量级。

另一个结构性发现：**state 的三种变体给出的 loss 完全相同**——该 π0.5 基座
（`discrete_state_input=False`）**不消费 proprioception**，只有图像/语言/动作进入前向。
所以 state 的符号/夹爪约定不影响模型，只需要保证动作列是逐帧增量。

落地方式：`scripts/make_robotwin_delta_actions.py` 把数据集里的 `actions` 列预先改写成
`a[t] − s[t]`，配置保持 `extra_delta_transform=False`。**不要**打开 openpi 的
`DeltaActions`：它减的是查询帧的 state（即上面的 `delta_now`，0.1949），语义不同。

验证：基座在官方数据上的 flow loss **0.3926 → 0.0739**（同一探针，5.3 倍下降）。

## 4. 当前状态与后续

已完成：代码分支 `feat/RoboTwin`（35 项 Con1 单测通过）、基座布局转换、数据 v2.1 转换、
动作约定判定与落地、数据管线端到端跑通（`scripts/probe_robotwin_base_loss.py`）。

后续：

1. 建 Con1 缓存：V-JEPA 目标（3 视角 × 1408）→ R tokens（64×2048）；
2. 两臂各 4500 步：A = Con1 livecross，B = 完整算法（Con1 + Con2 + 全前缀 VLM 上下文）；
3. 配对 flow 探针（含精确基座参考）出机制指标；闭环评测等仿真器准备好再做。
