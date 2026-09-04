# Con1：Action-Grounded Change CoFlow

> 状态：Phase A 最小验证实现；后续阶段在 Phase A 通过前不实施
>
> 分支：`feat/con`
>
> 基座：作者发布的 π0.5 JEPA-WAM，step `59999`
>
> 数据：只使用原始 LIBERO 专家轨迹，不采集干预或反事实动作数据

## 1. Con1 要解决的问题

JEPA-WAM 已经能够从当前观测预测 future representation，但原动作生成路径并不直接使用这项预测。更根本
的问题是，完整 current–future latent 同时包含大量静态场景、物体外观和任务阶段信息，预测准确不代表它
与动作生成对齐。

Con1 只解决一个问题：

> **能否从冻结 JEPA 中提取动作可辨识的变化表示，将其蒸馏成部署时只依赖当前观测的变化先验，再让变化
> 与动作在同一个 Flow 中共同生成？**

整体链条为：

```text
真实前后观测 → action-grounded change posterior B^R
当前观测和语言 → deployable change prior B^P
(action noise, B^P) → joint Action–Change Flow → (expert action, B^R)
```

这三步共同构成一个 Con1，不是三个独立贡献。

## 2. 数据与边界

每条训练样本来自同一条专家 episode：

```text
(o_t, language, q_t, A*_[t:t+H-1], o_(t+H)),  H=10
```

首轮只保留完整的 H10 片段，不跨 episode，也不混用原 JEPA-WAM 的约 `t+31` target。

π0.5 内部动作张量为 `[10,32]`；LIBERO 的真实控制量是前 7 维：

- π0.5 action-flow loss 继续使用原始 `[10,32]` 定义；
- Phase A 的逆动力学诊断只预测 `[10,7]`，不让 padding 零值降低 loss。

现有专家数据没有“同一状态执行多个动作”的反事实样本。因此 Con1 只能声称学习到
**observationally action-identifiable change**，不能声称辨识了严格因果动作后果。

## 3. Phase A：学习动作可辨识的变化后验

### 3.1 No-change-referenced latent displacement

冻结 V-JEPA，同时编码真实转移和无变化参考：

```text
Y^R_t = stopgrad(E_J([o_t, o_(t+H)]))  ∈ R^(576×1408)
Y^0_t = stopgrad(E_J([o_t, o_t]))      ∈ R^(576×1408)
```

沿用 JEPA cosine geometry，对每个 token 做无参数 L2 normalization：

```text
Y_bar = Y / max(||Y||_2, eps)
DeltaY_t = Y_bar^R_t - Y_bar^0_t
```

不对 `DeltaY_t` 再做整体归一化，因为其 token norm 表示相对 no-change reference 的变化强度：

```text
||DeltaY_i||_2^2 = 2 * (1 - cosine(Y^R_i, Y^0_i))
```

`DeltaY` 保留 `24×24` token 顺序，但它只是 **no-change-referenced semantic displacement**，不是光流、
像素对应或严格因果变化。

### 3.2 Change encoder 与 inverse decoder

第一版使用：

```text
K=16 change tokens
d_B=16 per-token dimensions
B^R_t = G_psi(DeltaY_t) ∈ R^(16×16)
```

`G_psi` 是两层 query readout：16 个 learned queries 通过 cross-attention 读取 576 个 `DeltaY` tokens，
hidden width 为 256，最后投影为 `[16,16]`。`K=32` 只作为主模型成立后的容量消融。

变化后验本身仍只读取 `DeltaY_t`。逆动力学头额外读取部署时天然可用的当前机器人状态 `q_t`，但不读取
当前图像或语言：

```text
A_hat_t = D_inv(B^R_t, q_t) ∈ R^(10×7)
L_inv = Huber(A_hat_t, A*_[...,0:7])
```

这样 `B^R` 的语义仍然是变化表示，而 decoder 回答的是“在当前机器人状态下，什么动作对应这种变化”。
若把 `q_t` 直接混入 `B^R` 的标签构造，`G_psi` 可能仅编码 state shortcut，因此首版不这样做。

Phase A 冻结 V-JEPA、π0.5 和 JEPA-WAM，只训练 `G_psi` 与 `D_inv`。

### 3.3 Phase A 究竟检验什么

Phase A 只检验：

> **在已经给定当前机器人状态后，JEPA latent displacement 是否仍为专家动作提供可泛化的增量信息？**

它不检验 rollout 成功率、不检验当前观测能否预测该变化，也不检验因果反事实。

验证集必须按完整 episode 隔离。四个模型使用相同的 `G_psi + D_inv` 结构、宽度、训练步数和随机种子；
被移除的输入以零张量替代，从而保持参数量一致：

| 输入 | 作用 |
|---|---|
| `q_t` | state-only 基线，测量当前状态能预测多少动作 |
| `DeltaY_t` | change-only 基线，测量变化本身包含多少动作信息 |
| `(q_t, DeltaY_t)` | 主方法，检验 change 在 state 之外的增量价值 |
| `(q_t, Y^R_t)` | raw-pair 对照，判断 latent change 是否优于完整 latent pair |

Phase A 的核心统计量是同一 held-out expert sample 上的配对 loss improvement：

```text
improvement_state = L_inv(q_t) - L_inv(q_t, DeltaY_t)
improvement_pair  = L_inv(q_t, Y^R_t) - L_inv(q_t, DeltaY_t)
```

继续 Phase B 的最低条件：

1. `(q_t,DeltaY_t)` 的 held-out loss 低于 state-only；
2. episode-level bootstrap 95% CI 下，`improvement_state` 仍大于 0；
3. `(q_t,DeltaY_t)` 优于 change-only，证明 state conditioning 确实必要；
4. `(q_t,DeltaY_t)` 的 held-out loss 低于 `(q_t,Y^R_t)`；
5. episode-level bootstrap 95% CI 下，`improvement_pair` 仍大于 0。

不再使用打乱 transition 的输入作为证据：打乱后的配对不来自专家数据分布，它只能证明 decoder 会响应
输入变化，不能证明 change 的语义或控制价值。

若条件 1–3 失败，说明 change 没有提供 state 之外的动作信息；若条件 4–5 失败，说明 no-change subtraction
相对 raw pair 没有优势。任一种情况都先停止，不进入 Phase B，也不在运行中修改算法。

## 4. Phase B：将未来后验蒸馏为当前变化先验

Phase A 完成后冻结 `G_psi` 和 `D_inv`：

```text
B^R_t = stopgrad(G_psi(DeltaY_t))
```

后续 Phase B 不使用外挂 Transformer。仿照 JEPA-WAM，在冻结 VLM 序列中加入 16 个 change-query tokens；
这些 query 可读取当前 image-language prefix 和原有 64 个 JEPA-WAM future-query states：

```text
[image, language, R^JEPA_(1:64), Q^B_(1:16)] → frozen VLM
H^B_t = hidden states at Q^B_(1:16)
B^P_t = Proj(AdaLN(H^B_t; E_q(q_t))) ∈ R^(16×16)
L_distill = Huber(B^P_t, B^R_t)
```

数学上的 `P_omega` 就是这 16 个 VLM 内生 queries、state conditioning 和轻量 projection，而不是第二个
预测网络。`B^R` 仍不含 state；`q_t` 只帮助 current-only student 预测专家未来变化。

这里才是严格意义上的蒸馏：

```text
future-privileged posterior B^R → current-only prior B^P
```

### Phase B 检验什么

Phase B 只检验：

> **训练时依赖未来观测得到的动作相关变化后验，能否从部署可用的当前观测与语言中被预测？**

在 episode-heldout 数据上比较：

- learned `B^P`；
- global mean `B^R`；
- task-conditioned mean `B^R`；
- 一个从训练开始就移除语言输入的 matched predictor。

同时通过冻结 `D_inv(B^P)` 报告动作诊断，但不把它称为策略成功率。

继续 Phase C 的最低条件：

1. learned prior 的 `L_distill` 优于 global mean 和 task mean；
2. 完整 predictor 优于 matched no-language predictor；
3. `D_inv(B^P_t,q_t)` 的 held-out action loss 优于 `D_inv(global/task mean,q_t)`。

若 learned prior 不超过 task mean，说明它主要记忆任务平均变化，停止 Con1。

## 5. Phase C：Action–Change CoFlow

Phase C 检验的不是“更多条件是否帮助动作”，而是：

> **让动作假设和变化假设在同一个生成过程中双向共同演化，是否优于把 `B^P` 当作固定 condition？**

沿用 OpenPI 的时间方向 `tau=1→0`。

动作路径：

```text
A_tau = tau * epsilon_A + (1-tau) * A*
V*_A  = epsilon_A - A*
```

变化路径：

```text
B_tau = tau * B^P + (1-tau) * B^R
V*_B  = B^P - B^R
```

二者共享同一个 `tau`：

```text
(V_hat_A, V_hat_B)
    = F_CoFlow(A_tau, B_tau, frozen_prefix, tau)

L = L_action + 0.1 * L_change
```

第一版只在冻结 Action Expert 倒数四层之前插入一个双向 CoFlow block：

```text
Action hidden → attend Change hidden → refined Action hidden
Change hidden → attend stopgrad(Action hidden) → refined Change hidden
```

`Action→Change` 乘固定的 `(1-tau)`：动作接近纯噪声时不应主导变化；不使用 learned gate 或硬阈值。
π0.5、VLM、V-JEPA、JEPA-WAM 和全部 Action Expert 在首版中冻结，只训练 CoFlow block。

### Phase C 检验什么

使用相同训练数据、steps 和新增参数预算比较：

1. frozen π0.5 JEPA-WAM；
2. fixed condition：`B^P→Action`，B 不演化；
3. independent flows：Action 与 Change 各自演化但不交互；
4. full CoFlow：Action 与 Change 双向交互。

首先只做 held-out offline flow matching，不启动模拟器。继续联合推理的最低条件：

```text
L_action(full CoFlow) < L_action(fixed condition)
L_action(full CoFlow) < L_action(independent flows)
L_change(full CoFlow) < L_change(independent flows)
```

若 full CoFlow 不超过 matched fixed condition，说明双向共同演化没有价值，Con1 在这里失败。

## 6. 联合推理与最小 rollout

只有 Phase A、B、C 全部通过，才接入原 π0.5 ODE solver。

部署输入只有：

```text
o_t, language, q_t
```

初始化：

```text
A_1 ~ N(0,I)
B_1 = P_omega(o_t, language)
```

联合积分：

```text
(A_1,B_1) → ... → (A_0,B_0)
```

最终只执行 `A_0[...,0:7]`。部署生成动作时不存在未来观测、`Y^R`、`Y^0` 或 V-JEPA teacher。

先运行一个固定 manifest 的 20-episode paired smoke test，只检查：

- 相同 seed 下是否比 frozen baseline 少失败；
- 是否出现明显策略退化或动作发散；
- 推理延迟是否可接受。

20 episodes 只用于淘汰错误方法，不用于论文结论。只有 smoke test 不退化且出现正向信号，才运行预注册的
LIBERO-Plus L4/L5 大规模评测。

## 7. 最小实现顺序

### Step 0：数据与结构测试

8 条样本，验证：

- H10 时间对齐与 episode 边界；
- `Y^R/Y^0/DeltaY: [B,576,1408]`；
- `B^R/B^P: [B,16,16]`；
- delta token norm 未被二次归一化；
- 推理接口没有 future observation 参数。

### Step 1：Phase A 快速证伪

```text
320 expert pairs
episode-level 160 train / 160 validation
500 steps, batch 32
```

同时训练 matched state-only、change-only 和 state+raw-pair controls。500 steps 不满足预设 held-out 增量条件
就停止；通过后才考虑延长到 2,000。

### Step 2：Phase B 快速证伪

```text
same episode split
500 steps, batch 32
```

不超过 task mean 就停止；有信号才延长到 2,000。

### Step 3：Phase C 快速证伪

```text
1,000 steps, batch 16–32
offline flow metrics only
```

不超过 fixed condition 就停止，不接 solver。

### Step 4：推理与小 rollout

先验证 zero-init baseline equivalence，再跑 20 个固定 paired episodes。通过后才扩大训练和评测。

## 8. 整个 Con1 的证据链

| 阶段 | 科学问题 | 成功证据 | 失败后动作 |
|---|---|---|---|
| Phase A | JEPA latent change 是否在 state 之外提供动作信息？ | state+change 显著优于 state-only/change-only，并优于 state+raw-pair | 停止 Con1 |
| Phase B | 未来后验能否蒸馏为当前先验？ | learned prior 优于 global/task mean，并依赖语言 | 停止 Con1 |
| Phase C | joint evolution 是否优于普通 condition？ | full CoFlow 优于 matched fixed condition 和 independent flows | 停止 Con1 |
| Inference | offline 改善能否进入动作生成？ | solver 稳定、20-episode paired smoke 出现正向信号 | 不做大评测 |
| Final | 是否真正改善 OOD 控制？ | 预注册 L4/L5 paired success 与统计检验 | 缩小或否定论文主张 |

这条证据链避免再次用“JEPA loss 下降”替代“动作和成功率改善”。

## 9. 可以与不能声称的内容

核心贡献表述：

> **We extract an action-identifiable latent displacement from a frozen JEPA using a no-change reference, distill
> its future-privileged posterior into a current-only change prior, and jointly evolve change and continuous action
> through a coupled flow model.**

可以声称：

- no-change-referenced latent displacement；
- inverse-dynamics-aligned change posterior；
- future-privileged posterior-to-prior distillation；
- joint Action–Change Flow。

不能声称：

- `B^R` 是严格因果动作后果；
- 16 个 token 对应固定物体或空间区域；
- 专家数据提供同状态多动作干预；
- inverse-action loss 自动保证 rollout 成功；
- Con1 已经包含 TTA/TTT 或持续学习。

## 10. 首版锁定项

1. 只使用专家数据，不采集 intervention；
2. `DeltaY = L2Norm(Y^R) - L2Norm(Y^0)`；
3. `K=16,d_B=16`，`K=32` 只做容量消融；
4. `B^R` 只由 `DeltaY` 构造；Phase A inverse decoder 额外读取 `q_t`，但不读取 image 或 language；
5. Phase B 才构成 posterior-to-prior distillation；
6. Phase C 只使用一个 CoFlow block；
7. 首版冻结完整基础模型和 Action Expert；
8. 不加入 tracker、point flow、VQ、MPC、额外 reward 或手工 gate；
9. 每个阶段未通过对应对照就停止，不用扩大训练掩盖失败；
10. Con2/TTT 等待 Con1 的控制证据成立后再设计。
