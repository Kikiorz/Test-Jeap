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

_待填：`summary_robotwin_random20_ft_<step>.tsv` 落地后粘贴。_

## 5. 怎么读

* **Clean 列**：模型在它训练分布内的任务上有没有退化。
* **Random 列**：这才是这个实验要动的量。
* 每格 25 个 episode，单任务噪声约 ±10pp，20 个任务合起来每个条件 500 个
  episode，均值噪声约 ±2pp——**结论要落在均值上，不要落在单任务上**。
* 和 π0.5 那一列的差异不只是数据（随机化 vs clean），还有数据量和配方，所以
  这张表证明的是「随机化数据训完能不能动 Random 这一列」，不是纯粹的
  「数据来源」消融。要做纯消融得再用 clean 数据按同一配方训一遍。
