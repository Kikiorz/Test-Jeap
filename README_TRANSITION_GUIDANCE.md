# 当前方案：JEPA Transition–Action Guidance

> 状态：`feat/point` 当前研究与实现合同（分支名仅为历史遗留）。  
> 基座：作者发布的 π0.5 JEPA-WAM step `59999`。  
> 数据：原始 LIBERO 专家轨迹；不使用 tracker、point flow、光流、成功图像或奖励。

## 1. 核心判断

JEPA 不必直接输出机器人动作。它适合从当前观测和语言提出一个未来 transition hypothesis；真正缺失的是
一个经过控制目标监督、可在部署时继续校准的 **transition→action guidance field**。

整条方法只有一个共享变量和一个共享映射：

```text
执行前：o_t,l ──JEPA-WAM──► predicted transition U^P
                                  │
                                  ▼
                        action guidance g_phi
                                  │
                                  ▼
                        frozen pi0.5 Action Flow ──► A_exec

执行后：[o_t,o_(t+H)] ──frozen V-JEPA──► achieved transition U^R
                                                │
                           known A_exec ─────────┘
                                                ▼
                                      update the same g_phi
```

因此离线学习与测试时适应不是两个外挂任务：二者都在学习同一个关系——**某个 JEPA transition 与哪段
机器人动作相容**。

## 2. 不使用 tracker 的 transition 定义

控制时间窗严格固定为当前 π0.5 的 `H=10`：

```text
(o_t, l, q_t, A*_[t:t+H-1], o_(t+H)).
```

执行前只能使用当前观测：

```text
R_t   = F_JW(o_t,l)
Y^P_t = G_align(R_t)                         # [B,24*24,1408]
U^P_t = Normalize(Pool_3x3(Y^P_t))           # [B,64,1408]
```

训练期或动作执行后，真实的前后观测走 V-JEPA 自己的 target encoder：

```text
Y^R_t = stopgrad(E_J([o_t,o_(t+H)]))         # [B,24*24,1408]
U^R_t = Normalize(Pool_3x3(Y^R_t))           # [B,64,1408]
```

`U^R` 不是人工标签，也不是 tracker 输出。它是 V-JEPA 对这个时间窗学习出的 joint transition
representation。`U^P` 与 `U^R` 使用完全相同的 pooling 和逐 token L2 normalization。

必须保持术语克制：

- `U` 可以叫 learned transition representation；
- 不能叫 point flow、optical flow 或真实物体位移；
- π0.5 的连续生成轨迹才叫 Action Flow。

## 3. Phase 0：时间尺度对齐

作者 checkpoint 的 JEPA future offset 约为 31 帧，而动作块只有 10 步。直接把二者配对会产生错误监督。
我们已经用冻结 V-JEPA 生成 `H=10` targets，并只适配 alignment output head：

```text
released H31 predictor, held-out relative gain over persistence: -7.20%
H10-aligned output head, held-out relative gain over persistence: +5.99%
matched-pair preference: 98.13% -> 99.38%
```

因此后续只使用 H10-aligned predictor；VLM 和 Action Expert 不在 Phase 0 重训。

## 4. Contribution 1：transition-conditioned action guidance field

### 4.1 为什么不再联合生成 transition

我们测试过从 `(Gaussian action noise,U^P)` 联合运输到 `(A*,U^R)` 的双流模型。修正 transition loss
尺度后，生成的 transition 已能接近匹配的 `U^R`，但动作 ODE 采样仍变差，且打乱 `U^P` 几乎不改变
动作。这说明 joint flow 可以把 transition 分支拟合好，却仍允许 action 分支忽略它。

当前方案不再为了“有 Flow”而生成第二条 latent flow。`U^P` 作为 JEPA plan 保持固定；新增模块直接学习
它应如何改变 π0.5 的动作向量场。

### 4.2 基础 Action Flow

沿用 π0.5 的约定：`tau=1` 为高斯噪声，`tau=0` 为专家动作。

```text
A_tau = tau * epsilon + (1-tau) * A*
v*_A  = epsilon - A*
v_pi  = frozen_pi05(A_tau,o_t,l,q_t,tau)
```

### 4.3 共享 guidance field

少量 action queries 读取 `8x8` transition tokens、当前 proprioception、`A_tau` 和 `tau`，输出动作速度
修正：

```text
g_phi = Guidance(A_tau, q_t, U, tau)
v     = v_pi + g_phi.
```

`g_phi` 在每个 ODE step 进入速度场，而不是在最终 action projection 后加一次 residual，也不重新训练
第二个 policy。第一版使用标准多头注意力加固定二维位置编码；方法创新在“JEPA transition 作为可适应的
flow guidance”，不把一个复杂 attention 结构包装成贡献。

离线只优化原 action flow target：

```text
L_guidance = E ||v_pi + g_phi(A_tau,q_t,U,tau) - v*_A||^2.
```

训练时以相同概率使用 `U^P` 和 stop-gradient `U^R`。前者匹配执行前推理分布；后者让同一 mapping 能在
动作执行后接受 achieved transition。VLM、V-JEPA、JEPA-WAM predictor 与 π0.5 Action Expert 均冻结。

零初始化 guidance output，保证 step 0 与作者 checkpoint 数值一致。为了证明不是普通额外容量，必须有：

```text
frozen pi0.5
matched-parameter guidance without U
guidance with shuffled U
full transition guidance
```

### 4.4 已验证的最小信息前提

在 40 个任务的 320 个 H10 片段上，按 episode 隔离为 160 train / 160 validation；一个相同容量的
inverse decoder 比较 `q_t` 与 `(q_t,U)`。三个随机种子结果为：

| seed | q-only MSE | + observed `U^R` | 相对下降 | + predicted `U^P` | 相对下降 |
|---:|---:|---:|---:|---:|---:|
| 31 | 0.2330 | 0.1444 | 38.0% | 0.1583 | 32.1% |
| 37 | 0.2438 | 0.1507 | 38.2% | 0.1707 | 30.0% |
| 43 | 0.2482 | 0.1550 | 37.6% | 0.1652 | 33.4% |

打乱 `U` 后 MSE 明显回升。这证明当前 JEPA transition 含有跨 episode 可泛化的 action imprint，而且
current-only prediction 保留了大部分该信息。它只验证了 guidance 的信息前提，尚未证明 rollout 成功率。

## 5. Contribution 2：用 achieved transition 更新同一 guidance

执行 `A_exec` 后自然得到下一观测，不需要任何标签生成器：

```text
U^R_ach = Normalize(Pool_3x3(E_J([o_t,o_(t+H)]))).
```

由于控制器知道自己发出的 `A_exec`，可重新采样 `epsilon,tau`，构造标准 flow-matching tuple：

```text
A_tau_exec = tau*epsilon + (1-tau)*A_exec
v_exec     = epsilon - A_exec
L_online   = ||v_pi(A_tau_exec) + g_phi(A_tau_exec,q_t,U^R_ach,tau) - v_exec||^2.
```

部署时只更新 `g_phi` 中 rank-4 transition K/V adapters；所有 backbone 和 π0.5 参数冻结。下一动作块仍用
执行前可得的 `U^P`，但其 transition→action coupling 已由真实交互校准。

这属于 reward-free inverse adaptation，而不是 RL：没有 reward、value、advantage 或 policy gradient。

### 5.1 为什么不能直接自我模仿

若一次错误动作也被无条件强化，online loss 下降不等于任务变好。因此离线训练采用 support/query
meta-alignment：

```text
phi'   = phi - eta * grad_phi L_online(support)
L_meta = L_expert_action(query; phi').
```

support 与 query 来自同一预定义 deployment condition 的不同时间窗。部署时只需 `L_online`；
meta-objective 只用于让无标签更新方向在训练期被“下一段专家动作是否更准”约束。这里没有手工 gate。

必须分别报告 episode-reset 与 persistent-session 协议；后者同时报告 clean LIBERO retention。

## 6. 方法的单一论文故事

```text
JEPA predicts WHAT transition should occur;
the guidance field maps that transition to HOW pi0.5 should move;
the achieved JEPA transition recalibrates the same mapping online.
```

这不是“Con1 用 JEPA、Con2 再挂 TTT”，而是一个 predict–act–observe–adapt 闭环：

```text
U^P --guide--> A_exec --environment--> U^R_ach --adapt same guide--> next A.
```

## 7. 最小后续验证

不立刻跑完整 LIBERO-Plus。按以下顺序停止/继续：

1. 在冻结 π0.5 上训练小型 `g_phi`，held-out active-7D flow loss 必须优于 matched no-U guidance；
2. 打乱 `U^P` 必须显著恶化，证明模型确实使用 JEPA transition；
3. 同一 10-step ODE 的离线 action MSE 必须改善，不能只看 velocity loss；
4. 通过后才在封存的 LIBERO-Plus L4/L5 小子集做 paired rollout；
5. Con1 超过作者 baseline 后，再验证一次 meta-aligned online update 是否改善 query action loss。

若第 3 项失败，就否定当前 guidance 参数化；不再通过增加 tracker、router、memory bank 或更多 attention
分支来掩盖失败。

## 8. 文献定位

- [JEPA-WAM](https://github.com/SpriteWithoutIce/openpi_jepawam) 提供 π0.5 checkpoint、future queries 与
  V-JEPA transition supervision；
- [On the Guidance of Flow Matching](https://arxiv.org/abs/2502.02150) 给出通用 flow-matching guidance
  框架与 training-based guidance field；
- [PAD](https://arxiv.org/abs/2007.04309) 证明部署时可用自监督辅助目标更新策略适配部分。

我们的待验证贡献不是这些单项模块，而是：用 JEPA learned transition 训练一个控制对齐的 action-flow
guidance，并用动作执行后自然获得的同一 transition 对该 guidance 进行 reward-free 在线校准。

## 9. 当前代码与证据

- `scripts/fit_horizon_alignment_head.py`：H10 predictor 时间尺度校准；
- `scripts/run_transition_inverse_gate.py`：无 tracker 的 transition→action 信息检验；
- `src/openpi/models/coflow.py` 与 `scripts/run_coflow_core_gate.py`：失败 joint-flow 假设的可复现原型；
- 新的 `g_phi` 尚未接入完整 π0.5；上表不能被写成 rollout 或最终 Con1 成果。

