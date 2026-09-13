# RoboTwin 2.0 · 二十任务随机化微调（π0.5）

目标：把 π0.5 直接在 RoboTwin 2.0 的**随机化（randomized）**演示上微调，然后在
同样的 20 个任务上测 Clean / Random，回答"官方那份数据没有随机化，随机化数据练完
到底能不能把 Random 那一列拉起来"。

## 1. 为什么必须自己造数据

`TianxingChen/RoboTwin2.0` 上的 LeRobot 发布包只有 **clean** 一份：

| 文件 | 内容 |
|---|---|
| `lerobot_dataset/RoboTwin_lerobot_v21.zip` (3.9 GB) | 2500 episodes / 548,985 帧 / 50 任务 × 50 条 clean，3 相机 480×640，15 fps |
| `lerobot_dataset/RoboTwin_lerobot_v30.zip` (6.7 GB) | 同上，v3 布局 + 内联图 |
| `dataset/<task>/<robot>_randomized_500.zip` | **唯一的随机化来源**，每任务 500 条，HDF5 + mp4 |

发布包里**没有 Random 划分**，也没有任务身份（原始 parquet 只有
`state/action/timestamp/frame/episode/index/task_index`，`task_index` 索引的是 2410 条
指令字符串）。所以 PACE 表里 π0 的 `Ran. 23.9` 是**只用 clean 数据训练**的模型的成绩；
要改这一列只能自己去按任务的 zip 里取随机化数据。

## 2. 动作/状态约定（对着发布包验证过，不是猜的）

用 `RoboTwin_lerobot_v21/data/chunk-000/episode_000000.parquet` 与原始 HDF5 对比：

| 项 | 结论 | 证据 |
|---|---|---|
| `observation.state[t]` | `joint_action/vector[t]`（14 = 6+1+6+1） | parquet 与原始 HDF5 逐帧 `allclose == True` |
| `action[t]` | `joint_action/vector[t+1]`，末帧重复 | `\|a[t] − s[t+1]\|` 均值 **0.0**；而 `\|a[t] − s[t]\|` = 0.0088 |
| fps | 15 | `timestamp[1] = 0.0666…` |
| 相机 | `head_camera→cam_high`、`left_camera→cam_left_wrist`、`right_camera→cam_right_wrist` | RoboTwin `_base_task.update_wrist_camera(left_camera, right_camera)`；发布包同名列 |
| 列名 | 保留发布包的 `action`（单数） | 本配置用 `action_key="action"` + `action_sequence_keys=("action",)` |

原始帧是 320×240 JPEG（内嵌在 HDF5 的 `observation/*_camera/rgb` 字节流里），
发布包是把它放大到 480×640，信息量没有增加；训练本来就要缩到 224，所以直接存
224×224 JPEG，省一半磁盘。

## 3. 数据集

`scripts/build_robotwin_random20.sh` + `scripts/build_robotwin_random_dataset.py`：

* 20 个任务 × 200 条随机化 episode（从每任务的 500 条里取前 200 条）；
* 每条 episode 从 `instructions/episodeN.json` 的 `seen`+`unseen` 里按固定种子抽 1 条指令
  （评测默认用 `unseen`，两边都要覆盖）；
* 直接 `zipfile` 读成员、只解码需要的 HDF5，**从不解包到磁盘**；一条任务转换完就删掉归档，
  峰值磁盘 ≈ 6 GB；
* 输出 LeRobot **v2.1**，帧以 JPEG 字节内联（`meta/info.json` 里 `dtype: image`，
  parquet schema metadata 里写 HF 的 `Image` 特征），训练不需要解视频；
* `--append` 支持一条一条累积：episode 编号、任务表、统计量都会接着上一次续。

规模：20 × 200 = 4000 episodes / ≈57 万帧（与发布包 54.9 万帧同量级），约 15 GB。

## 4. 训练

配置 `pi05_robotwin_random20_ft`（`src/openpi/training/config.py`）：

| 项 | 值 | 说明 |
|---|---|---|
| 基座 | `/workspace/models/pi05_base/params` | 官方 π0.5 base，未被 RoboTwin 见过 |
| 数据 | `/workspace/data/robotwin_random20_inline` | 上面的随机化集 |
| 动作块 | `action_horizon=16` | 与本分支其它 RoboTwin 配置一致；推理端从 checkpoint 读回 |
| batch / 卡 | 128 / 4×GPU，`fsdp_devices=4`，48 个 loader worker | 实测 **3.7 s/step**，4 卡利用率 100%（不是数据瓶颈） |
| lr | cosine，warmup 500，peak **5e-5** → 5e-6 | 步数比官方少一个量级，用更高峰值补偿 |
| 步数 | 10,000（≈10.3 h） | 官方 `pi05_base_aloha_full_sim_*` 用 60k×256；我们按实测步时定 |
| checkpoint | 每 2000 步，只保留最新 | 单个 ckpt 31 GB（12 GB 参数 + 19 GB 优化器状态），磁盘只有 ~50 GB |

命令：

```bash
bash scripts/build_robotwin_random20.sh                        # 造数据
python scripts/compute_norm_stats.py --config-name=pi05_robotwin_random20_ft
bash scripts/run_robotwin_random_ft.sh                         # 训练
```

## 5. 评测

RoboTwin 仿真（SAPIEN + Vulkan，本机 `.102` 已验证 Vulkan 可用）跑 20 任务 ×
Clean/Random，与 PACE 表同口径对比。参考点：官方 cotrain（clean-only）π0 = 62.5 / 23.9。

评测链路（`scripts/robotwin_eval_suite.sh`）：

1. 起一个策略服务器（`XPolicyLab/setup_policy_server.py`，加载我们的 ckpt 与 norm stats）；
2. 每个任务 × {`demo_clean`(seen), `demo_randomized`(unseen)} 调
   `scripts/eval_policy_xpolicylab.py --eval_batch true --num_workers 8`，
   8 个仿真并行、共享同一个服务器；
3. 结果写进 `eval_result/.../_result.txt`，同时汇总成 `summary_<run>_<step>.tsv`。

`scripts/robotwin_ft_then_eval.sh` 会等训练进程退出，取最新 checkpoint，再自动跑完整套。

### 5.1 这台机器上踩过的坑（都已解决，写下来省得重装）

| 问题 | 处理 |
|---|---|
| `assets/*.zip` 没下载 | `assets/_download.py`（objects/embodiments/background_texture ≈15 GB），解压后删 zip |
| `curobo_left.yml` 不存在 | 必须跑 `scripts/update_embodiment_config_path.py`，它把 `*_tmp.yml` 里的 `${ASSETS_PATH}` 展开成真实路径 |
| curobo 编译不过 | 机器只有 CUDA 13.2，torch 是 cu121；装 `cuda-toolkit-12-8` + torch 2.8.0+cu128（RTX PRO 6000 是 sm_120，cu121 的 torch 连 `torch.sign` 都跑不了），再 `CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0` 编译 curobo |
| `warp.torch` 没了 | curobo 0.7.8 用旧 API，`scripts/patch_curobo_warp_api.py` 把它换成 `wp.device_from_torch` |
| 评测要 conda | 两个官方 shell 脚本用 `conda info --base` 起服务器/客户端；这里直接按它们展开的命令跑（见 suite 脚本），不装 conda |
| 客户端 obs 键名 | 训练数据是 `observation/image` 扁平键，仿真客户端发的是 `{"images": {"cam_high": ...}}`；`robotwin_policy.RoboTwinInputs` 现在两种都认 |
| 磁盘 | 单 checkpoint 31 GB（12 GB 参数 + 19 GB 优化器态），只保留最新；为此清掉了 LIBERO-Plus 的中间数据（可从 HF 重下） |
