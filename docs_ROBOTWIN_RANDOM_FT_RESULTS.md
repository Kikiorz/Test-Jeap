# RoboTwin 2.0 随机化微调：结果

这一页只放结果和怎么读它。实验怎么搭的、踩了什么坑，都在
`docs_ROBOTWIN_RANDOM_FT.md`。

## 1. 我们到底在回答什么问题

RoboTwin 2.0 的官方发布包只有 **clean** 演示；PACE 表里 π0.5 的
`Ran. 36.7` 是**只用 clean 数据训练**的模型在随机化场景下的成绩。
所以这个实验回答的是：

> 直接用 20 个任务的**随机化**演示去微调 π0.5，能不能把 Random 那一列拉起来？

## 2. 设置

| 项 | 值 |
|---|---|
| 基座 | 官方 π0.5 base（未被 RoboTwin 见过） |
| 数据 | 自建 RoboTwin 2.0 random 集：20 任务 × 200 条随机化 episode ≈ 4000 条 / 57 万帧（15 GB，LeRobot v2.1） |
| 训练 | batch 128，FSDP 4 卡，cosine lr 峰值 5e-5，10000 步，action_horizon 16 |
| 评测 | 20 任务 × {`demo_clean`(seen), `demo_randomized`(unseen)} × 25 episodes，**官方 expert check 打开** |
| 参考点 | π0.5（clean 训练）75.1 / 36.7；PACE（Ours）87.1 / 48.5 |

## 3. 训练曲线

两次进程拼起来的一万步（中间因 `/dev/shm` 写满在 step 6000 崩过一次，从
step 2000 带原优化器状态续训，见 `docs_ROBOTWIN_RANDOM_FT.md` §5.4）：

| step | 2000 | 3000 | 4000 | 5000 | 6000 |
|---|---:|---:|---:|---:|---:|
| loss（第一段） | 0.0072 | 0.0053 | 0.0049 | 0.0039 | 0.0037 |
| loss（续训后） | 0.0072 | 0.0054 | 0.0048 | — | — |

两段在相同 step 上的 loss 不可逐步对比：续训后 dataloader 重新洗牌，看到的是不同的
batch 序列。要看的是趋势——**到 6000 步仍在下降**。

## 4. 闭环结果

评测跑在仿真机上（`expert_check` 需要 curobo，训练机没有，理由见
`docs_ROBOTWIN_RANDOM_FT.md` §5.5）。链条跑完后这张表由
`scripts/robotwin_eval_report.py` 直接输出：

**结果快照**：`results/summary_robotwin_random20_ft_9999.tsv`（30/40 个配置）。

评测跑到 **36/40** 时两台机器被回收（连接全部拒绝），所以下表的 4 个任务缺席；
快照里完整的 30 行如下（数值为 25 episodes 的成功率）：

| 任务 | 我们 Clean | π0.5 Clean | 我们 Random | π0.5 Random |
|---|---:|---:|---:|---:|
| Adjust Bottle | 32 | 97 | 32 | 26 |
| Beat Block Hammer | 16 | 76 | 8 | 9 |
| Click Alarmclock | 52 | 90 | 56 | 63 |
| Click Bell | 52 | 98 | 48 | 58 |
| Dump Bin Bigbin | 0 | 95 | 8 | 42 |
| Grab Roller | 24 | 92 | 16 | 64 |
| Handover Mic | 8 | 84 | 16 | 8 |
| Lift Pot | 20 | 63 | 8 | 4 |
| Place Bread Basket | 0 | 51 | — | 27 |
| Place Bread Skillet | 0 | 56 | 4 | 20 |
| Place Burger Fries | 0 | 83 | 4 | 54 |
| Place Cans Plasticbox | 0 | 36 | — | 42 |
| Place Empty Cup | 4 | 74 | 0 | 59 |
| Place Object Basket | 16 | 66 | 12 | 8 |
| Place Shoe | 16 | 29 | 12 | 15 |
| Press Stapler | 56 | 67 | **76** | 22 |
| Shake Bottle Horizontally | 未跑 | 100 | 未跑 | 61 |
| Shake Bottle | 未跑 | 99 | 未跑 | 82 |
| Stack Bowls Three | 未跑 | 59 | 未跑 | 29 |
| Stack Bowls Two | 未跑 | 87 | 未跑 | 40 |

**配对均值（只统计我们有数的任务）**

| | 我们 | 论文 π0.5 | 差 |
|---|---:|---:|---:|
| Clean（16 任务） | **18.5** | 72.3 | **−53.8** |
| Random（14 任务） | **21.5** | 32.3 | **−10.8** |

结论：**这个 10k 步的随机化微调没有把 Random 列拉起来**。个别任务（`press_stapler`
+54pp、`handover_mic` +8pp、`place_object_basket` +4pp）确实超过 π0.5，但
`dump_bin_bigbin`（−34）、`grab_roller`（−48）、`place_burger_fries`（−50）
把均值拉到 −10.8pp；Clean 更是差 53.8pp。

**两点必须写进结论的限制**：

1. **缺席的 4 个任务恰好是 π0.5 在 Random 上最强的**（82/61/40/29），所以
   −10.8pp 这个数不能当作最终定论，但方向上不可能翻盘（需要它们把 21.5 拉到
   32.3 以上，即平均要超过 60%）。
2. 同一配置重跑两次的结果不一样（`place_object_basket` random 跑出过 3/25 和
   9/25，`place_empty_cup` clean 跑出过 0/25、1/25、4/25），说明**除了二项噪声
   之外还有运行间波动**，单任务差值（±10pp 量级）不要单独解读。

## 5. 怎么读

## 4b. 并行度：渲染是瓶颈，不是越多越好（2026-09-13 夜）

第一次起评测用了 `WORKERS=12`，结果**卡死**：客户端日志 15 分钟只打了 1 条
`step:`，被 `svulkan2 OIDN Error: invalid handle` 刷屏，`ENV_GPU` 显存涨到
**96.8 GB / 97.9 GB**——12 个渲染器把一张卡塞满，仿真几乎不前进。同一台机器上
LIBERO sweep 的 32 个 shard 还在争 GPU 2。

改成 `WORKERS=8` + 把渲染挪到**空闲的 GPU 3**（`POLICY_GPU=1`、`ENV_GPU=3`）后：
`adjust_bottle/demo_randomized` **25 个 episode 9 分钟跑完**，显存 66 GB。

结论：这套仿真里 SAPIEN 的光追渲染是硬瓶颈，并行度得按**显存**算（每个渲染器
约 8 GB），不是按 CPU 核数算。另外 suite 是可续跑的（已完成配置按
`<task>_<cfg>_<step>.txt` 跳过），所以中途换参数重启不会丢已完成的行。

* **Clean 列**：模型在它训练分布内的任务上有没有退化。
* **Random 列**：这才是这个实验要动的量。
* 每格 25 个 episode，单任务噪声约 ±10pp，20 个任务合起来每个条件 500 个
  episode，均值噪声约 ±2pp——**结论要落在均值上，不要落在单任务上**。
* 和 π0.5 那一列的差异不只是数据（随机化 vs clean），还有数据量和配方，所以
  这张表证明的是「随机化数据训完能不能动 Random 这一列」，不是纯粹的
  「数据来源」消融。要做纯消融得再用 clean 数据按同一配方训一遍。
