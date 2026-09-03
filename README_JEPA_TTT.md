# 当前方法：Control-Aligned JEPA Test-Time Training

> 状态：`feat/point` 当前主线与最小实验合同。  
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

## 9. 实现顺序

1. 在缓存特征和冻结 π0.5 velocity 上验证 support→query 的一步更新方向；
2. 在 `embed_prefix` 的 image-token 路径加入 opt-in low-rank adapter，并验证零初始化等价；
3. 实现一阶/二阶一步 meta-objective，先用单卡小 batch；
4. gate 通过后再用四卡训练；
5. 测试时只保存小 adapter state，不覆盖作者 checkpoint。

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

