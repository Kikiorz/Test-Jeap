# RoboTwin 2.0 · 二十任务随机化微调（π0.5）

目标：把 π0.5 直接在 RoboTwin 2.0 的**随机化（randomized）**演示上微调，然后在
同样的 20 个任务上测 Clean / Random，回答"官方那份数据没有随机化，随机化数据练完
到底能不能把 Random 那一列拉起来"。

## 0. 从零复现清单（2026-09-14，两台机器已回收）

先说结论：这一轮做完 10k 步微调 + 30/40 配置的闭环评测，**Random 列没有超过 π0.5**
（−10.8pp），Clean 差 53.8pp，判断是**欠训练**。下次直接从 30k 步起步。按下面顺序做，
坑都已经踩过：

1. **机器**（两台最省时间）
   * 训练机：4×sm_120 卡；`/dev/shm` 要 **≥120G**（一次存档 31G + 数据集 21G + 基座
     12G + HF 缓存 22G），根盘 ≥40G。`/dev/shm` 若 noexec，venv 必须放根盘。
   * **评测机必须能跑 curobo**：RoboTwin 每个 seed 都先跑专家演示（`expert_check`
     默认开），专家演示走 curobo 的 `plan_grippers`。没有 curobo 只能关掉专家检查，
     那就换了协议、和 PACE 表不可比（`--expert_check false` 只能是最后手段）。
     编译 curobo：装与 torch 匹配的 CUDA toolkit，`TORCH_CUDA_ARCH_LIST=12.0`。
   * 评测机还要 Vulkan：`apt install libvulkan1 mesa-vulkan-drivers vulkan-tools`，
     `vulkaninfo` 能报 1.3。
2. **数据**：`bash scripts/build_robotwin_random20.sh`（20 任务 × 200 条随机化演示，
   LeRobot v2.1，约 15GB），再 `python scripts/compute_norm_stats.py
   --config-name=pi05_robotwin_random20_ft`。
3. **训练**：`STEPS=30000 bash scripts/run_robotwin_random_ft.sh`（基座 = 官方 π0.5
   base）。两条硬约束：
   * `HF_HOME` **不要指根盘**：`load_dataset` 会重建 arrow 缓存（20G+），根盘必炸。
   * 存档算术：一次 = 12G params + 19G train_state = **31G**，orbax 先写后删。
     **前两次存档前要 ≥62G 空闲**（那时没有旧存档可退），之后每次 ≥31G 即可。
     写满的后果不是丢一次存档，而是**两小时后 `wait_until_finished` 抛异常、把训练
     进程一起带走**（§5.4）。
4. **评测**：`bash scripts/robotwin_eval_suite.sh`，20 任务 × {`demo_clean`(seen),
   `demo_randomized`(unseen)} × 25 episodes。**并行度按显存算**：每个 SAPIEN 渲染器
   约 8GB，`WORKERS ≈ 单卡显存/8` 再留余量（12 个渲染器会把 97.9G 的卡打到 96.8G，
   然后卡死）；**policy server 要和渲染放不同卡**。suite 可续跑：已完成的配置按
   `<task>_<cfg>_<step>.txt` 跳过，改参数重启不丢行。
5. **出表**：`python scripts/robotwin_eval_report.py <summary.tsv>`，自动和论文 π0.5
   那一列做配对均值对照（只统计跑完的任务）。
6. **两个已知不稳定点**：低成功率配置每局跑满 400–700 步预算（慢 4 倍是正常的，不是
   挂死）；真正挂死的表现是 worker 变僵尸 + client 停在 `poll` + 四卡 0%——单配置
   `timeout 45min` 兜底，超时记 FAILED，事后单独重跑。

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

### 5.2 训练/评测在两台机器上的布局

**当前实际安排（2026-09-13）**：微调在 `.21`（`154.59.156.21:45968`）跑，评测
**就地**在 `.21` 跑（不占 `.102`，`.102` 上是你自己的 LIBERO-Plus 微调）。

| 项 | 位置 | 说明 |
|---|---|---|
| 训练 repo | `/dev/shm/rt_ft/ws`（`/workspace/robotwin_ws` 软链） | 分支 `feat/Robotwin-ran-ft`，venv 在根盘 `/opt/venv-robotwin`（shm noexec） |
| checkpoint | `/dev/shm/rt_ft/ckpt/pi05_robotwin_random20_ft/robotwin_random20_ft/<step>` | 每 2000 步存一次，只留最新 |
| 仿真 venv | `/opt/rt-eval`（根盘，2.0 GB） | 基解释器来自训练 venv，用 `_train_venv.pth` 把训练 venv 的 site-packages 接进来，再补 `sapien 3.0.0b1 / mplib 0.2.1 / toppra / open3d / msgpack-numpy / websockets / h5py / opencv-headless`。**同一个解释器同时当 policy server 和仿真客户端** |
| 仿真代码+资产 | `/dev/shm/robotwin/code` | 资产是数据，可以放 noexec 的 shm；Meshes 16 GB 在评测链里才从 `.102` 拉 |
| 结果 | `/dev/shm/rt_eval_results/summary_<exp>_<step>.tsv` | 每任务两行：`demo_clean`(seen) / `demo_randomized`(unseen) |

评测链：`scripts/robotwin_ft_then_eval_21.sh`（`bash /root/rt_chain.sh`），
已挂在 `.21` 上。它按顺序做：等训练进程退出 → 删掉非最终 checkpoint 和
`train_state` 腾 shm → 从 `.102` rsync 仿真代码+资产 → 展开
`*_embodiment_tmp.yml` 的 `${ASSETS_PATH}` → 先跑 1 任务 1 episode 的 smoke →
再跑 20 任务 × {clean, random} × 25 episodes（8 个并行仿真）。smoke 失败就直接
停，不浪费后面的 40 次客户端调用。

**开跑前的预检（都已实测，不是假设）**：

### 5.3 磁盘事故与处置（2026-09-13，step 4000）

`/dev/shm` 被训练 checkpoint 填满（251G/251G，剩 896M），**step 4000 的存档直接
写失败**：

```
ValueError: RESOURCE_EXHAUSTED: ... Failed to write to file
[OS error 28: ENOSPC No space left on device]
```

训练本身没崩（异步存档失败只丢那一次存档），但如果不腾空间，**后面每次存档都会
失败，包括最后 10000 步那次**——那样评测链就只能拿到 2000 步的 checkpoint。

存档的算术：一次存档 = 12G params + 19G train_state = **31G**，而 orbax
`max_to_keep=1` 是「先写新的、再删旧的」。所以稳定条件不是「腾一次」，而是
**每次存档前必须有 ≥31G 空闲，并且上一个完整 checkpoint（31G）还在**，这样删旧
的时候才能把空间退回来（`keep_period=100000` 大于总步数，等于只留最新一个）。

处置（挑的都是**可再生成**的，没动用户数据）：

| 动作 | 释放 | 理由 |
|---|---:|---|
| 删 `4000.orbax-checkpoint-tmp-0` | 22G | 写失败的半截目录，orbax 不认它 |
| 删 `/dev/shm/robotwin` | 15G | 仿真代码+下载残留，评测链会从 `.102` 重新 rsync |
| **没动** `/dev/shm/ts_jepa` | 132G | 旧 Con1 RoboTwin 特征缓存，删了要重算，属于要用户拍板的一次性操作 |

结果：`/dev/shm` 从 896M 空闲回到 **37G**，够 6000 / 8000 / 10000 三次存档
（每次写完删掉上一个，收支平衡）。

### 5.4 训练在 step 6000 崩掉，以及从 2000 续训（2026-09-13）

5.3 的处置**不够**：真正致命的不是那次写失败本身，而是它**两小时后才炸**。
orbax 的异步存档在 09:48 失败后，主线程在 11:49:06 走到 `wait_until_finished`
时把这个异常重新抛出来：

```
[E] [process=0][thread=MainThread][step=4000][wait_until_finished] Save Finalize thread (save_finalize) failed.
  File ".../orbax/checkpoint/checkpoint_manager.py", line 1315, in save
ValueError: RESOURCE_EXHAUSTED: ... [OS error 28: ENOSPC No space left on device]
```

训练进程直接退出，GPU 归零。**这时盘上只有 step 2000 的 `params`（12G），它的
`train_state` 已被 orbax 清理掉**——也就是无法原地 `--resume`。

顺带暴露的第二个问题：链条把「训练进程消失」当成「训练跑完了」，直接拿 step
2000 去跑评测（smoke 因为仿真还没 staging 而失败）。已修复：链条现在要求
**step ≥ `EXPECT_MIN_STEP`（默认 9999）且 `params/manifest.ocdbt` 存在**，
否则报错退出，不会拿半截/陈旧 checkpoint 去评测。

**恢复过程**：

1. `.102` 上还留着搬运时用的**完整 step 2000**（`params` 12G + `train_state`
   19G）——把它 rsync 回 `.21`（19G @ 14MB/s ≈ 23 分钟）。
2. 删 `/dev/shm/rt_ft/models/pi05_base`（12G）：`--resume` 时
   `init_train_state(resume=True)` 根本不会走 weight loader，所以基座权重对续训
   无用，需要时可以从 HF 重下。
3. `RESUME=1` 重启，从 step 2000 续，优化器状态完整，学习率调度也接着原曲线。

**踩到的坑（第二次启动失败）**：`HF_HOME` 必须是 `/dev/shm/rt_ft/hf_home`。
第一次重启用了 `/workspace/.hf_home`（在根盘，只剩 6.5G），`load_dataset("parquet", ...)`
要重建 arrow 缓存，直接 `OSError: [Errno 28] No space left on device`。用回
`/dev/shm/rt_ft/hf_home`（22G，缓存命中）后 30 秒内进入训练。

正确的重启命令（在 `/dev/shm/rt_ft/ws` 下）：

```bash
ROOT=/dev/shm/rt_ft/ws PYTHON=/opt/venv-robotwin/bin/python CKPT_DIR=/dev/shm/rt_ft/ckpt \
  EXP=robotwin_random20_ft STEPS=10000 BATCH=128 FSDP=4 NUM_WORKERS=48 SAVE_INTERVAL=2000 \
  RESUME=1 NCCL_PRELOAD=/opt/venv-jepa/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2 \
  LOG=/dev/shm/rt_ft/train.log HF_HOME=/dev/shm/rt_ft/hf_home \
  bash scripts/run_robotwin_random_ft.sh
```

代价：step 2000→6000 的 4000 步白跑，从 2000 重新走，总计 8000 步 ≈ 8 小时。

### 5.5 评测改到仿真机上跑（专家检查需要 curobo）

「就地评测」这条路在 `.21` 上**跑不通**，而且是在 smoke 里抓到的，不是上线才发现：

1. 补 `open3d` 和 RoboTwin 的资产之后，客户端能完整导入；
2. 但没有 curobo 时 `envs/robot/planner.py` 里 `CuroboPlanner` 根本没定义，
   `robot.py` 第一行 import 就炸——仓库里的 fallback 补丁（追加一个占位类）
   能解决这一层；
3. 真正的墙在下一层：RoboTwin **每个 seed 都要先跑一遍专家演示**
   （`expert_check`，默认开），专家演示走 `robot.left_plan_grippers` →
   curobo 的 `plan_grippers`。占位类只能让 import 过去，跑不出演示：

```
error occurs during expert check! seed=100041 err=AttributeError:
curobo is not installed, so planner.plan_grippers is unavailable
RuntimeError: Batch eval completed zero episodes. skipped_seeds=50
```

注意**策略本身不需要 curobo**：`_base_task.take_action(action_type='qpos')`
用的是 mplib 的 TOPP（`left_mplib_planner.TOPP`）。所以 curobo 只在「专家检查」
这一层被需要，而那一层决定了每个 seed 算不算有效 seed——关掉它
（`--expert_check false`）就等于换了协议，和论文那一列没法直接比。

三条路里选了第三条：

| 方案 | 结论 |
|---|---|
| `.21` 上 `--expert_check false` | 能跑，但协议和 PACE 表不一致，**不采用** |
| `.21` 上装 curobo | 要匹配 torch 的 CUDA toolkit（torch 是 cu126，curobo 要 cu128 编译）+ sm_120 编译，根盘只剩 6.5G，**不可行** |
| **仿真机上跑评测** | `.102` 的 eval venv 里 curobo 能 import，192 核负载 ~9，每张卡 76G 空闲，**采用** |

评测机上的分工：策略服务器用 **GPU 1**、仿真渲染用 **GPU 2**（这两张是四张里
利用率最低的），`.102` 上的 LIBERO 训练不动。链条
`scripts/robotwin_ft_then_eval_split.sh` 负责：等训练退出 → 校验 step ≥ 9999 且
`params/manifest.ocdbt` 在 → 删掉 19G 的 `train_state` → rsync 最终 `params`
到 `.102` → 专家检查按官方协议跑 1 任务 1 episode 的 smoke → 通过后跑满
20 任务 × clean/random × 25 episodes → 拉回结果 + 出 PACE 对照表。

**已经用 step 2000 把这条远端命令实测了一遍**（就是链条里那条命令）：服务器在
GPU 1 起、两个配置都出结果写盘，`adjust_bottle` clean 0/1、random 1/1。
按这个速度，40 个配置 ≈ 4–5 小时。

### 5.6 如果要接着训（拉长步数）

> **2026-09-14 状态**：两台机器（`.21` 训练机、`.102` 仿真机）已被回收，连接全部
> 拒绝。于是：仿真环境、`/dev/shm` 里的数据集/缓存/checkpoint、以及
> `train_state`（续训能力）**全部不可回收**；能保留的是
> `results/summary_robotwin_random20_ft_9999.tsv`（30/40 配置）和本仓库的代码 +
> 这些文档。要继续做 RoboTwin，需要新机器上重走 `docs_ROBOTWIN_RANDOM_FT.md`
> §3–§5（造数据 → 训练 → 评测），下面的续训命令只在**还留着 checkpoint 的机器**
> 上有意义。

最终 checkpoint 的 `train_state` **不要删**（评测只读 `params`，但删了就只能从基座
重训）。链条现在保留它。续训命令（在 `/dev/shm/rt_ft/ws` 下）：

```bash
RESUME=1 STEPS=30000 ROOT=/dev/shm/rt_ft/ws PYTHON=/opt/venv-robotwin/bin/python \
  CKPT_DIR=/dev/shm/rt_ft/ckpt EXP=robotwin_random20_ft BATCH=128 FSDP=4 NUM_WORKERS=48 \
  SAVE_INTERVAL=2000 \
  NCCL_PRELOAD=/opt/venv-jepa/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2 \
  LOG=/dev/shm/rt_ft/train.log HF_HOME=/dev/shm/rt_ft/hf_home \
  bash scripts/run_robotwin_random_ft.sh
```

注意两点：

* **这是热重启**：`config.lr_schedule` 是按 `num_train_steps` 重建的余弦，把
  10000 改成 30000 后续训，学习率会从「1 万步余弦的末端」跳回「3 万步余弦在
  第 1 万步的值」——比原来高。想避免跳变就得自己换 schedule。
* 存档节奏不变：每次 31G，写完删上一个，`/dev/shm` 里 48G 空闲刚好够周转。

链条也据此加固：等训练退出后先等 `*.orbax-checkpoint-tmp-*` 消失，再取最大
step，并**要求该 step 目录下有 `params/`**，否则直接报错退出，不会拿半截
checkpoint 去评测。

| 检查 | 结果 |
|---|---|
| `/dev/shm` 能否执行 venv | **否**（`micromamba: Permission denied`），所以 venv 必须放根盘；容器没有 `CAP_SYS_ADMIN`，不能给 shm 加 `exec` |
| 根盘剩余 | 32 GB 总，当前剩 6.6 GB（`/opt/rt-eval` 2.0 GB） |
| Vulkan | `apt install mesa-vulkan-drivers` 后 `vulkaninfo` 报 1.3.275；SAPIEN 离屏渲染实测出图（空场景 mean=0.0） |
| policy server | 用 step 2000 起服务，**端口 5 秒内开始 listen** |
| 仿真客户端 | `eval_policy_xpolicylab.py` 全模块导入通过（补完 `open3d` 和 `assets/*.json` 之后） |

两台机器都在跑实验，卡不共享；`.21` 的 `/dev/shm` 是 **noexec**，所以凡是需要执行
`.so` 的东西（venv、curobo、sapien）只能放根盘，数据/checkpoint 放 `/dev/shm`，用软链
把配置里写死的路径接上：

| 路径 | 指向 | 位置 |
|---|---|---|
| `/workspace/robotwin_ws` | `/dev/shm/rt_ft/ws`（openpi 仓库） | 源码在 shm，venv 除外 |
| `/opt/venv-robotwin` | — | 训练用 venv（`UV_PROJECT_ENVIRONMENT`，根盘，可执行） |
| `/workspace/data/robotwin_random20_inline` | `/dev/shm/rt_ft/data/...` | 数据集 21 GB |
| `/workspace/models/pi05_base` | `/dev/shm/rt_ft/models/pi05_base` | 基座 12 GB |
| checkpoint | `/dev/shm/rt_ft/ckpt` | 31 GB/个，只留最新 |

跨机搬运用 `.21 → .102` 的直连 ssh（把 `.21` 的公钥加进 `.102` 的
`authorized_keys`），rsync 单流约 15 MB/s，三路并行约 45 MB/s：数据集 21 GB、
checkpoint 31 GB、基座 12 GB，合计约 1 小时。搬完用
`RESUME=1 PYTHON=/opt/venv-robotwin/bin/python bash scripts/run_robotwin_random_ft.sh`
从最新 checkpoint 继续，不丢已训的步数。

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
