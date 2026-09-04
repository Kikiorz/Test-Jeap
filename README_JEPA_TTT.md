# 当前研究状态：JEPA–Action 接口的核心验证

> **2026-09-04 更新：目前没有一种接口通过完整控制 gate，因此尚未进入 LIBERO-Plus rollout。**
> `feat/point` 已依次验证 CoFlow、共享 adapter TTT、inverse-TTT 和 transition–action energy；这些实验均
> 使用作者 π0.5 JEPA-WAM step `59999`、H=10 对齐数据以及 episode-disjoint 划分。当前结论是
> **JEPA transition 含动作信息，但现有接口不能把该信息稳定转化为更好的 π0.5 动作。**

## 最新核心结果

| 验证 | 主要结果 | 判定 |
|---|---:|---|
| transition→action 信息量 | current-only predicted transition 的同任务 4-way action retrieval top-1 `81.88%`；no-change `40.63%` | `R` 确有动作相关信息 |
| 共享 image-adapter JEPA-TTT | correct update 与 within-task shuffled update 几乎相同；仅 `7/16` 同时优于两个控制项 | 无 target specificity，停止 |
| JEPA+inverse adapter TTT | correct 仅 `8/24` 同时优于 no-change、shuffled-action、shuffled-transition | 无 target specificity，停止 |
| transition-energy 局部梯度 | 人工扰动专家动作上 `76.88%` 改善；对真实 π0.5 action 仅 `60.63%` 改善，MSE `0.00473339→0.00473073` | 方向弱，bootstrap 95% CI 跨 0 |
| transition-energy 4-candidate rerank | 原 energy：MSE `0.004733→0.008319` | 明显失败 |
| policy hard-negative energy | rerank MSE `0.004733→0.007711` | 失败 |
| policy listwise-ranking energy | rerank MSE `0.004733→0.006342` | 仍失败 |

因此不能把“retrieval 准确”写成“控制有效”，也不能因为 spatial shuffle 会恶化就运行昂贵 rollout。当前
最小科学结论是：

```text
R contains action information
        ≠
R provides a usable local control objective for a strong pi0.5 policy.
```

下一步若继续，不再尝试 `R→Action` 的直接条件、残差、adapter 或无状态 compatibility。唯一仍合理的
根本修改是显式拆开：

```text
JEPA-WAM current-only prediction  → desired transition U_des
current state + candidate action  → predicted consequence U_hat(A)
action selection/refinement       → match U_hat(A) to U_des under the pi0.5 prior
executed action + next observation→ update the same consequence model
```

该方向必须先通过 action-sensitivity gate：改变候选 action 时，`U_hat(A)` 必须产生与真实 transition 一致的
可区分变化；否则不再扩展结构。

# 历史方法：Control-Aligned JEPA Test-Time Training

> 状态：`feat/point` 当前主线与最小实验合同；代码已实现，核心 gate 尚未通过。
> 基座：作者发布的 π0.5 JEPA-WAM checkpoint，step `59999`。  
> 约束：不使用 tracker、point flow、光流、成功图像、环境奖励或测试集动作标签。  
> 分支名 `feat/point` 仅为历史遗留；本文中的表示不是几何 point flow。

## 1. 问题

JEPA-WAM 在训练时让 current-only future queries 预测真实 current–future V-JEPA 表征，但默认动作 token
看不到这些 future queries。已有实验进一步表明，即使把 transition 表征直接接到动作侧，强 π0.5 也可以
忽略它：

```text
JEPA loss 下降  ≠  action loss 下降  ≠  rollout 成功率提高
```

因此本文不再假设“更准确的 JEPA latent 天然会改善动作”。研究问题改为：

> **能否离线学习一个共享的适配空间，使部署时仅通过真实前后观测计算的 JEPA prediction loss 更新该空间，
> 并使这一步无标签更新在下一状态上改善 π0.5 动作？**

## 2. 不使用 tracker 的自监督信号

控制窗口与动作块严格对齐为 `H=10`：

```text
(o_t, language, q_t, A_[t:t+H-1], o_(t+H)).
```

执行前，JEPA-WAM 只从当前观测和语言预测 transition：

```text
U^P_t = Normalize(Pool_8(G_align(F_JW(o_t, language))))
```

执行动作后，冻结 V-JEPA 从自然出现的前后观测构造 target：

```text
U^R_t = stopgrad(Normalize(Pool_8(E_J([o_t, o_(t+H)]))))
```

两者均为 `[B,64,1408]`，使用相同 `24×24 → 8×8` pooling 和逐 token L2 normalization。

```text
L_JEPA = mean(1 - cosine(U^P_t, U^R_t)).
```

`U^R` 是 V-JEPA 学到的 joint transition representation，不是人工标签。它不应被称为 point flow、
optical flow、真实物体位移或成功目标。

部署因果顺序为：

```text
o_t → generate/execute A_t → observe o_(t+H) → compute L_JEPA → update → generate A_(t+H)
```

未来观测从不用于生成已经执行的 `A_t`。

## 3. 唯一新增对象：共享测试时适配参数

在 current image tokens 进入 PaliGemma prefix 前加入一个小型低秩 residual adapter：

```text
h'_img = h_img + W_up σ(W_down LN(h_img)),       rank r = 4 or 8.
```

记其参数为 `phi`。这个位置有意位于两条路径的共同上游：

```text
adapted image prefix ──► future queries ──► U^P ──► L_JEPA
                    └──► frozen Action Expert ──► action flow
```

因此同一个 `phi` 同时影响 JEPA prediction 和动作生成。与旧方案不同：

- 不把高维 `U` 当作新的动作 condition；
- 不增加第二个 world model 或 latent flow；
- 不用 inverse decoder 直接替代 π0.5；
- 不给 action velocity 叠加一个独立 residual head；
- 不更新 V-JEPA target encoder。

π0.5、SigLIP/PaliGemma 主干、future queries、alignment head 和 Action Expert 首轮全部冻结；测试时只更新
`phi`。低秩不是论文贡献，而是约束在线更新容量和成本。

## 4. 核心贡献：让 JEPA 更新方向受到动作目标约束

普通 TTT 直接执行：

```text
phi' = phi - eta * grad_phi L_JEPA.
```

但这只保证 representation prediction 变好。我们的训练目标直接评价更新后的动作：

### Support transition

从一条专家轨迹中取已发生的 transition：

```text
S = (o_s, o_(s+H)).
phi_S = phi - eta * grad_phi L_JEPA(S; phi).
```

### Query action

从相同 deployment condition 的后续窗口取：

```text
Q = (o_q, language_q, q_q, A*_q),   q > s.
```

用更新后的 `phi_S` 计算原 π0.5 flow-matching loss：

```text
L_outer = L_action(Q; phi_S).
```

离线优化：

```text
min_phi E_[S,Q] L_action(Q; phi - eta grad_phi L_JEPA(S;phi))
        + lambda_ret L_action(Q; phi).
```

`lambda_ret` 只用于保持零更新性能；第一版不加入 gate、router、memory bank、progress head 或 reward model。
核心科学约束是：

```text
Δ_JEPA = -grad_phi L_JEPA
必须经过训练，使其成为降低后续 action loss 的方向。
```

这一步借鉴测试时训练和梯度元学习的成熟原则，但将 inner objective 固定为 JEPA-WAM 已有的未来表征预测，
将 outer objective 固定为 π0.5 原始动作目标。

## 5. 部署时持续学习

部署时不再需要 outer action label：

```text
1. 用 phi_t 和当前观测生成并执行动作块 A_t；
2. 得到自然出现的 o_(t+H)；
3. 计算 U^P_t 与 frozen U^R_t；
4. phi_(t+H) = phi_t - eta grad_phi L_JEPA；
5. 下一动作块由更新后的共享 prefix 生成。
```

需要分别报告：

- `episode-reset`：每个 episode 从离线学到的 `phi_0` 开始；
- `persistent-session`：跨 episode 累积，并同时报告 clean LIBERO retention。

这属于 reward-free TTT/TTA，不是 RL。机器人没有判断刚才动作是否成功；它只用未来观测校正自己的
transition prediction。更新是否有利于控制由离线 outer action objective 保证，而不是由手工 accept gate 保证。

## 6. Con1 与 Con2 是同一个机制

若论文仍用两项贡献表述：

1. **Control-Aligned JEPA Adaptation Objective**：离线学习 `phi_0` 与 JEPA inner gradient，使一次
   transition-prediction update 改善后续动作；
2. **Online JEPA Adaptation**：部署时用真实下一观测计算同一个 inner loss，持续更新同一个 `phi`。

更推荐在论文中把它们合并成一个主要贡献：

> **We meta-align JEPA-WAM's naturally available future-representation prediction gradient with the downstream
> action-flow objective, turning the auxiliary JEPA loss into a reward-free online policy adaptation signal.**

统一闭环为：

```text
predict transition → act → observe realized transition
        ↑                                  │
        └──── update the shared adapter ───┘
```

## 7. 当前证据与已经否定的路线

### 7.1 成立的信息前提

在 40 个 LIBERO 任务、320 个 H10 片段、episode-disjoint 划分上：

| inverse 输入 | held-out action MSE |
|---|---:|
| proprioception only | 0.2331 |
| no-change V-JEPA pair `[o_t,o_t]` | 0.1686 |
| current-only predicted `U^P` | 0.1564 |
| real transition `U^R` | **0.1460** |
| within-task shuffled `U^R` | 0.2212 |

真实 transition 相比 no-change 降低约 13.4%，同任务内打乱后明显恶化。这说明 JEPA transition 含有超越
静态场景和任务身份的动作相关时间信息。

### 7.2 已否定的接口

| 参数化 | 关键结果 | 决定 |
|---|---|---|
| Joint Action–Transition CoFlow | velocity loss 下降，但 integrated active-action MSE 从 0.4381 升至 0.4721；shuffle 几乎无影响 | 停止 |
| additive transition guidance | frozen π0.5 active MSE 0.01452，加入 guidance 后 0.01470；shuffle delta 约 `1.3e-6` | 停止 |
| inverse-proposal candidate reranking | first candidate MSE 0.00473，transition rerank 0.00548；shuffled transition 反而 0.00521 | 停止 |

这些结果共同说明：`U` 有动作信息，但精度不足以直接替代或修正强 π0.5。新方法因此不再直接解码、重排
或残差注入 `U`，而把 JEPA 用作共享策略表征的在线学习目标。

### 7.3 当前 control-aligned TTT 证据

共享 image-token adapter、一步 JEPA inner update 和一阶 meta outer objective 已经接入完整 π0.5 forward。
结构测试共 `6/6` 通过，包括关闭功能时的兼容性、零初始化输出等价性和 adapter 梯度。

第一轮极小 gate 使用 120 个 meta steps 和 16 个 episode-heldout support–query pairs；每个 query 当时只取
一个 flow time/noise draw：

| 初始化 | support target | mean query delta | median query delta | improved pairs |
|---|---|---:|---:|---:|
| 未做 meta-alignment | correct transition | `-2.61e-5` | `+3.18e-6` | `8/16` |
| meta-aligned | correct transition | `-1.86e-4` | `-4.41e-6` | `9/16` |
| meta-aligned | no-change | `-3.26e-5` | `+7.40e-6` | `7/16` |
| meta-aligned | within-task shuffled | `-9.86e-5` | `-0.65e-6` | `9/16` |

meta-alignment 后，correct update 相对 no-change 在 `10/16` 对上更好，相对 shuffled 在 `11/16` 对上更好，
但只有 `9/16` 同时优于两个控制项。均值还受到少量高 flow-loss draw 支配。因此这只是**弱的目标特异性
迹象，不构成方法有效证据**。

当前唯一追加实验不是增加模块，而是修正估计量：训练时轮换同一状态的独立 flow draws，验证时对四个
flow time/noise draws 求期望。若正确 transition 在这个更稳定的动作目标上仍不能一致优于 no-change 和
within-task shuffle，则当前 adapter 位置/一阶 meta-gradient 假设被否定，不进入 rollout。

该追加实验已经完成。240 个 meta steps、16 个 held-out pairs、每对 4 个 flow draws 的结果为：

| support target | mean query delta | median query delta | improved pairs |
|---|---:|---:|---:|
| correct transition | `-5.43e-5` | `-5.14e-6` | `10/16` |
| no-change | `-0.98e-5` | `-6.22e-6` | `10/16` |
| within-task shuffled | `-5.06e-5` | `-5.87e-6` | `10/16` |

correct 与 shuffled 的均值只相差 `-3.76e-6`；correct 只在 `7/16` 对上同时优于两个控制项。因此更稳定的
估计否定了当前版本：**一阶 meta-adapter 学到了一般性的小幅 action finetuning，而没有学会利用本次真实
transition。** 本版本不进入 rollout。

下一项最小实验回到同一个 JEPA 变量，但让 support loss 同时要求 transition 能重建已执行动作：

```text
L_online = L_JEPA(U^P, U^R) + lambda * L_inverse(I(U^P, q_t), A_executed).
```

`U^R` 仍由执行后 V-JEPA 产生，`A_executed` 是机器人本来就知道的动作；没有 tracker 或新标签。冻结的
inverse readout 只用于验证 action-decodable JEPA gradient 是否比纯 prediction gradient 更具控制特异性，
不能在 gate 通过前被包装成正式贡献。

## 8. 最小可证伪实验

正式训练与 rollout 前只做一个 gate：

```text
support: observed H10 transition, only L_JEPA available
query:   later state from the same held-out deployment condition
metric:  query action-flow loss before vs after exactly one JEPA step
```

比较：

1. frozen π0.5；
2. naive JEPA-TTT：未做 outer meta-alignment；
3. control-aligned JEPA-TTT；
4. shuffled `o_(t+H)` support；
5. no-change `[o_t,o_t]` support。

继续条件：

```text
L_action_query(aligned, after) < L_action_query(aligned, before)
L_action_query(aligned, after) < L_action_query(naive, after)
correct support update < shuffled/no-change support update
```

三项必须在 episode-heldout 数据和至少三个随机种子上成立。若失败，说明 JEPA prediction gradient 在这个
共享参数位置仍不具备可迁移的控制信息；不能靠增加 gate 或 attention 分支掩盖。

通过后才进行：

- LIBERO-Plus L4/L5 小规模 paired rollout；
- 扩展到完整 LIBERO-Plus；
- 第二个 benchmark 与 persistent continual adaptation。

## 9. 当前实现与下一步

已完成：

1. 在 `embed_prefix` 的 image-token 路径加入 opt-in low-rank adapter；
2. 实现 current-only JEPA prediction、真实 H10 V-JEPA target 和一步在线更新；
3. 实现一阶一步 meta-objective，并在完整冻结 π0.5 forward 上运行；
4. 保存独立 adapter state，不覆盖作者 checkpoint；
5. 保留 correct/no-change/within-task-shuffle 三种严格支持信号。

尚未完成：

1. 纯 JEPA 的 flow-draw averaged gate 已失败；
2. 尚无证据授权进行 LIBERO-Plus rollout；
3. 正在验证 JEPA prediction 与 executed-action reconstruction 的单一联合 online loss；
4. 该联合 loss 若仍无 target specificity，则停止当前共享 adapter 路线；不通过增加 gate、tracker 或更多
   attention 分支补救。

缓存实验只是验证更新机制，不构成最终动作改进结论。最终证据必须来自完整 π0.5 forward 和闭环 rollout。

## 10. 文献边界

- [JEPA-WAM](https://arxiv.org/abs/2608.09381)：提供 current-only future queries、V-JEPA target 与 π0.5
  checkpoint；
- [Self-Supervised Policy Adaptation during Deployment](https://arxiv.org/abs/2007.04309)：验证共享策略表征可用
  inverse-dynamics 等自监督任务进行部署适应；
- [VITA](https://openreview.net/pdf?id=V35oo1SVGH)：验证自监督测试时更新应通过下游预测目标进行梯度元对齐；
- [VANE](https://arxiv.org/abs/2608.09448)：说明 VLA future-representation TTT 的在线更新必须接受后续证据审计。

我们的边界不是“首次将 TTT 用于机器人”或“首次预测未来表征”，而是：

> **在 JEPA-WAM π0.5 中，针对已经实证存在的 JEPA–action disconnect，学习一个由真实 transition
> prediction error 驱动、并由 action-flow objective 元对齐的共享在线更新方向。**
