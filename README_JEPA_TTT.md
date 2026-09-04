# CoFlow-JEPA：当前方法、核心证据与下一步

> 分支：`feat/point`
>
> 基座：作者发布的 π0.5 JEPA-WAM，step `59999`
>
> 更新日期：2026-09-04
>
> 当前状态：**完整方法尚未获得控制有效性证据，暂不进行 LIBERO-Plus 大规模 rollout。**

## 1. 我们真正研究的问题

JEPA-WAM 已经能仅根据当前观测和语言预测一个未来转移表征：

```text
U^P_t = f_JW(o_t, language).
```

但作者默认不让 action tokens 读取 future queries。因此，JEPA 主要作为训练期辅助目标：

```text
JEPA prediction ──► auxiliary loss
JEPA prediction ──X──► action generation
```

我们此前已经验证，直接把该表征通过 condition、残差、adapter、deformable readout 或候选重排交给动作侧，
均不能稳定改善强 π0.5。核心问题不是“动作网络没有看到更多特征”，而是：

> **JEPA 表征能够描述任务转移，但未必表达当前动作假设与该转移之间的对应关系。**

因此最新方法不再采用固定的 `JEPA latent → Action` 接口，而研究：

```text
预测转移先验 → 转移与动作共同形成 → 执行
      ↑                              ↓
      └──── 用真实动作印记在线校准 ────┘
```

暂定方法名为 **CoFlow-JEPA — Joint Transition–Action Flow with Online Sparse Inverse Adaptation**。

## 2. 变量定义与严格边界

一次动作块长度固定为 `H=10`：

```text
(o_t, language, q_t, A*_[t:t+H-1], o_(t+H)).
```

执行前，JEPA-WAM 根据当前观测产生 desired transition prior：

```text
U^P_t = Normalize(Pool_8(G_align(R_t))) ∈ R^(64×1408).
```

训练或执行后，冻结 V-JEPA 根据真实前后观测产生 realized transition：

```text
U^R_t = stopgrad(Normalize(Pool_8(E_J([o_t, o_(t+H)])))) ∈ R^(64×1408).
```

两者必须使用完全相同的 V-JEPA 特征空间和 `24×24 → 8×8` pooling。`U^R` 是学习式 joint
current–future representation，而不是 tracker、光流、人工 point flow、成功图像或真实因果状态。

部署时生成当前动作只允许使用 `o_t`：未来观测 `o_(t+H)` 只能在动作已经执行之后用于下一次更新。

## 3. Con1：Prior-to-Outcome Action–Transition Co-Flow

### 3.1 核心思想

不是把固定 `U^P_t` 当作动作条件，而是让动作假设和转移假设在同一个 rectified-flow 过程中共同演化：

```text
(Gaussian action noise, predicted transition prior)
                         ↓ joint vector field
(expert action, action-consistent realized transition).
```

沿用 π0.5 的时间约定：`tau=1` 为 source，`tau=0` 为 target。

```text
A_tau = tau * eps_A + (1-tau) * A*
U_tau = tau * U^P   + (1-tau) * U^R

V*_A = eps_A - A*
V*_U = U^P - U^R
```

联合向量场为：

```text
[V_hat_A, V_hat_U] = F_theta(A_tau, U_tau, prefix(o_t, language, q_t), tau)

L_CoFlow = L_action + lambda_U * L_transition.
```

推理从 `(eps_A, U^P)` 开始联合积分，最终得到可执行动作 `A_0` 与 action-consistent transition
`U_0`。推理不使用真实未来、V-JEPA teacher 或额外环境 rollout。

### 3.2 最小网络原则

正文候选采用双流 joint-attention block，而不是再串联 world model：

- action stream 和 transition stream 保留独立 Norm、QKV 与 FFN；
- joint attention 允许两条流交换信息；
- transition loss 读取 action hidden 时使用 stop-gradient，避免辅助目标破坏动作表示；
- `Action → Transition` 耦合按 `1-tau` 连续增强，纯噪声阶段不污染 transition；
- VLM、V-JEPA 与大部分 Action Expert 首轮冻结。

这里唯一的结构主张是 **joint evolution**。DCT、tracker、视觉伺服、MPC、候选规划和手工 gate 均不加入
正文方法。

## 4. Con2：Action-Imprint Sparse Inverse TTT

动作块执行并获得 `o_(t+H)` 后，机器人天然知道一组无奖励事实：

```text
(U^R_t, A_exec_t).
```

Con2 从 `U^R_t` 中读取能够重建已执行动作的稀疏 action-imprint support：

```text
z_k = sum_i entmax(q_k K(U^R_i)) * V(U^R_i)
A_hat_inv = D_inv(z, q_t)
L_inv = Huber(A_hat_inv, A_exec).
```

Inverse readout 与 Con1 的 `Transition → Action` 路径共享 transition key/value projections。测试时只在这些
共享投影上更新低秩参数，其余策略、JEPA-WAM 和 V-JEPA 全部冻结：

```text
generate A_t → execute → observe U^R_t
             → one inverse update on shared coupling
             → generate A_(t+H).
```

这不是 RL：没有 reward、value、advantage 或 policy gradient。它是在线 inverse-dynamics adaptation。
由于“更会重建错误动作”不保证下一个动作更好，最终版本必须离线 meta-align 一次 inverse update，使它在同一
deployment condition 的 query state 上降低 action-flow loss。该步骤只有在 Con1 先通过控制 gate 后才实现。

## 5. 已完成的核心 falsification

### 5.1 历史直接接口均失败

| 接口 | 核心结果 | 判定 |
|---|---:|---|
| 原 Joint CoFlow gate | integrated action MSE `0.4381 → 0.4721`；shuffle 几乎无影响 | 当前表示下失败 |
| additive guidance | MSE `0.01452 → 0.01470` | 失败 |
| inverse candidate rerank | first `0.004733`；rerank `0.005481` | 失败 |
| shared-adapter JEPA-TTT | correct 与 shuffled transition 更新几乎相同 | 无 target specificity |
| inverse-TTT adapter | correct 仅 `8/24` 同时优于三个控制项 | 失败 |
| transition-energy gradient | `0.00473339 → 0.00473073`，bootstrap 95% CI 跨 0 | 不可靠 |
| energy rerank（原始/硬负样本/listwise） | `0.008319 / 0.007711 / 0.006342` | 均差于 first candidate |

这些负结果共同说明：

```text
R contains action information
        ≠
R is already a usable local control objective.
```

### 5.2 单轨迹相关性不等于动作后果

在普通专家轨迹上，real transition 可明显改善 inverse action reconstruction：

| 输入 | held-out action MSE |
|---|---:|
| proprioception only | `0.2331` |
| no-change pair | `0.1686` |
| current-only predicted transition | `0.1564` |
| real transition | **`0.1460`** |
| within-task shuffled real transition | `0.2212` |

同任务 4-way retrieval 中，real transition 达到 `88.13%`，current-only prediction 达到 `81.88%`，随机为
`25%`。但每个状态只有一条专家动作，因此网络可能利用“轨迹阶段 ↔ 动作”的相关性，而不是学习同状态下
局部 action effect。后续同状态实验验证了这一风险。

## 6. 最新同状态干预实验

### 6.1 实验合同

对 4 个 LIBERO-Spatial 任务各取 4 个初始状态。每个 MuJoCo 状态精确恢复后执行五个动作分支：

```text
base policy action, +x, -x, +y, -y.
```

共 `16 states × 5 branches = 80` 条转移。所有分支共享相同起点；恢复前图像最大 MAE 仅
`0.00729/255`，因此结果不是 reset 随机性造成的。

与此前四个相近 policy samples 相比：

| 候选 | action pair RMS | future-image pair MAE | realized V-JEPA pair distance |
|---|---:|---:|---:|
| 相近 policy samples | `0.0224` | `0.777` | `0.00598` |
| `base/±x/±y` | `0.0994` | `2.053` | `0.02088` |

### 6.2 V-JEPA 能否表示动作造成的差异？

在 8 个训练状态上学习一个极小 action-conditioned consequence model，并在另外 8 个完整 held-out 状态上
匹配五种真实 transition：

```text
top-1 = 70.0%, chance = 20.0%.
```

这说明当动作差异足够大时，冻结 V-JEPA realized transition **确实包含可辨识的 action imprint**。因此
不能简单下结论说“V-JEPA 不适合动作”。更准确的结论是：它对强 π0.5 周围非常接近的候选变化太不敏感，
但对较大的局部控制干预有信号。

### 6.3 现成 JEPA-WAM prior 是否可作为 10-step desired transition？

答案目前是否定的。

| prior | 到 base-action `U^R` 的距离 | persistence 到 base-action `U^R` |
|---|---:|---:|
| 作者 `t+31` head | `0.15553` | **`0.05189`** |
| H=10，仅重训 output projection | `0.16854` | **`0.05189`** |
| H=10，重训完整 alignment head | 约 `0.1662`（五分支均值） | **`0.05205`** |

H=10 alignment head 并非完全没学到东西：在独立的 320 个普通专家片段、episode-disjoint 验证上，仅重训
output projection 后 `prediction_to_matched=0.13293`，优于 persistence `0.14331`，相对改善 `5.99%`；
完整 head 的相对改善为 `5.27%`。但该收益不能迁移到同状态干预初始状态。

所以问题不是继续调 projection 的学习率，而是：

> **原 future-query hidden 本身是在 `t+31` 辅助目标下形成的；只换末端映射无法可靠得到控制窗口 H=10
> 的 desired transition prior。**

### 6.4 learned consequence selection 是否超过简单方向偏置？

使用完全相同的 consequence-model 参数，只替换 desired prior：

| desired prior | first/base | learned selection | best fixed direction |
|---|---:|---:|---:|
| 作者 `t+31` | `0.155945` | `0.154799` | **`+y: 0.153506`** |
| H=10 output-head | `0.169073` | `0.167037` | **`+y: 0.166330`** |

原 prior 的 learned selector 在 8 个 held-out 状态中全部选择 `+x`；H=10 版本也主要选择 `+x/+y`。两者
虽然相对 base 有很小的 latent-distance 改善，却都没有超过最佳固定方向。因此这不是状态相关规划能力。

即使把真实 base-action transition 当作 oracle desired target，同一模型也只在 `5/8=62.5%` 状态选回 base
action。说明 action↔transition 映射已有信号，但目前数据量和表示仍不足以可靠反演。

## 7. 当前问题定位

现在可以把失败拆成三个清楚、互不混淆的问题：

1. **尺度问题**：强 π0.5 的相近候选只产生约 `0.006` 的 V-JEPA 差异，容易被场景共享信息淹没；较大轴向
   干预达到 `0.0209` 后才可辨识。
2. **先验时间/语义错位**：作者 current-only prior 针对约 `t+31`，不能作为 10-step action chunk 的
   desired transition；仅重训 alignment head 不够。
3. **局部逆映射仍不稳**：控制轴上 action sensitivity 明显高于随机，但 oracle desired 只能 `62.5%`
   找回对应动作，不能承担在线控制。

因此目前不能声称 CoFlow-JEPA 有效，也不能进入 Con2 或 LIBERO-Plus rollout。但核心假设尚未被彻底否定：

```text
realized JEPA transition contains an action imprint,
but the released current-only prior and current coupling are not control-aligned.
```

## 8. 唯一允许的下一步：重学 H=10 future queries

下一步不再增加 attention、gate 或 reranker。只做一个根本修正：从作者 checkpoint warm-start，冻结 VLM、
V-JEPA 和 Action Expert，重新训练：

```text
64 future queries + JEPA alignment head
```

使其预测严格 H=10 的专家 realized transition。为避免再次只学任务进度，训练期加入一个共享、training-only
inverse objective，要求预测/真实 transition 都能恢复同一专家动作：

```text
L_phase0 = d(U^P_H10, stopgrad(U^R_H10))
         + lambda_inv * Huber(D_inv(U^P_H10), A*)
         + lambda_inv * Huber(D_inv(U^R_H10), A*).
```

这不是新的部署模块；inverse head 只约束 future queries 学到 action-decodable transition，并为后续 Con2
提供同一监督语义。首轮只做小批量、短训练，不修改 Action Expert。

继续到 CoFlow 的条件必须同时满足：

1. episode-heldout `U^P_H10` 稳定优于 persistence；
2. 同状态干预上，`U^P_H10` 不再塌缩成固定 `+x/+y` 偏置；
3. action-conditioned consequence 在 oracle desired 下显著超过 `62.5%`；
4. 基于 predicted desired 的选择超过最佳固定方向，而不只是超过 base；
5. shuffled action/transition 明显破坏以上结果。

若重训 future queries 后仍失败，则 released V-JEPA joint target 不适合作为 π0.5 的局部动作控制变量；届时应
停止 CoFlow-JEPA，而不是继续堆模块。

## 9. 实现与实验文件

核心诊断代码：

- `examples/libero/collect_action_interventions.py`：同状态多动作干预；
- `scripts/audit_openpi_jepawam_pi05.py`：冻结 V-JEPA target 与 JEPA-WAM prediction 审计；
- `scripts/fit_horizon_alignment_head.py`：H=10 alignment-head falsification；
- `src/openpi/models/action_consequence.py`：最小 action-conditioned consequence gate；
- `scripts/run_intervention_consequence_gate.py`：held-out action sensitivity、oracle 与 fixed-direction 比较。

远程关键产物：

```text
/workspace/artifacts/outputs/action_interventions_axis_spatial4x4_k5_d02_s233.npz
/workspace/artifacts/outputs/action_interventions_axis_spatial4x4_k5_d02_s233_teacher/
/workspace/artifacts/outputs/action_interventions_axis_spatial4x4_k5_d02_s233_predict/
/workspace/artifacts/outputs/action_interventions_axis_spatial4x4_k5_d02_s233_predict_h10/
/workspace/artifacts/outputs/intervention_axis_query_update_gate_s239/
/workspace/artifacts/outputs/intervention_axis_query_update_h10_fixed_gate_s239/
/workspace/artifacts/outputs/intervention_axis_query_update_oracle_desired_gate_s239/
```

## 10. 可以与不能声称的结论

目前可以声称：

> 在严格同状态干预下，冻结 V-JEPA transition 能辨识较大的动作后果，但作者 JEPA-WAM 的 current-only
> prior 与 π0.5 的 10-step 控制窗口不对齐；简单 horizon-head 重映射和已有 transition→action 接口均不足以
> 将该信号转化为状态相关动作选择。

目前不能声称：

- CoFlow-JEPA 提高了任务成功率；
- V-JEPA transition 是因果 point flow；
- latent distance 改善等价于控制改善；
- fixed-axis 改善等价于 learned planning；
- Con2 已具备部署时持续提升能力；
- 当前小样本 gate 等价于完整 LIBERO-Plus 结果。

完整论文故事仍保持为一条统一链：

```text
H=10 action-grounded transition prior
        → joint action–transition generation
        → execute and observe realized transition
        → sparse inverse update of the same coupling.
```

但每一箭头都必须先通过对应的可证伪实验，不能用结构复杂度代替控制证据。
