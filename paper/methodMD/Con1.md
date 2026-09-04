# Con1：Action-Grounded Change CoFlow

> 状态：Phase A 已完成选择，正在进行 A+B 最小联合验证
>
> 分支：`feat/con`
>
> 基座：作者发布的 π0.5 JEPA-WAM step `59999`
>
> 数据：原始 LIBERO 专家轨迹；不采集干预或反事实动作

## 1. 核心问题

JEPA-WAM 能够预测 future representation，但完整 latent 包含大量静态外观和场景信息，而且原动作路径在
推理时不使用这项预测。Con1 研究：

> **能否先从真实 JEPA transition 中提取动作可辨识的变化目标，再把该未来特权知识蒸馏进一个由当前观测
> 条件化的联合 Flow，使 latent change 与连续动作一起生成？**

训练链条：

```text
(o_t, o_t+H) ──frozen V-JEPA──► DeltaY_t ──Phase A──► B_t^R

(o_t, language, q_t) ──frozen VLM──► H_t^B ──────────────┐
                                                          ▼
                           (epsilon_A, epsilon_B) ── joint Action–Change Flow
                                                          ▼
                                                  (A_t*, B_t^R)
```

部署时：

```text
(o_t, language, q_t, epsilon_A, epsilon_B)
                         ──► (A_hat_t, B_t^P)
```

未来观测、V-JEPA teacher、`B_t^R` 和 inverse decoder 全部只在训练期存在。

## 2. 数据与时间尺度

每条样本来自同一个专家 episode：

```text
(o_t, language, q_t, A*_[t:t+H-1], o_(t+H)), H=10
```

- future frame 不跨 episode；
- `H=10` 与当前动作块物理跨度一致；
- Phase A 和最小联合验证使用相同的 episode-disjoint `160 train / 160 validation`；
- LIBERO 实际动作使用 `[10,7]`，不让 padding 维度稀释诊断指标。

## 3. Phase A：定义未来特权变化目标

### 3.1 No-change-referenced JEPA displacement

冻结 V-JEPA，同时编码真实转移和无变化参考：

```text
Y_t^R = stopgrad(E_J([o_t, o_(t+H)])) ∈ R^(576×1408)
Y_t^0 = stopgrad(E_J([o_t, o_t]))     ∈ R^(576×1408)
```

逐 token L2 normalization 后相减：

```text
DeltaY_t = L2Norm(Y_t^R) - L2Norm(Y_t^0)
```

不再对 `DeltaY_t` 整体归一化，因为 token norm 本身表示相对于 no-change reference 的变化强度。

### 3.2 选定的 change bottleneck

最终方法只有一个输入：

```text
B_t^R = G_psi(DeltaY_t) ∈ R^(16×16)
A_hat_t = D_inv(B_t^R)  ∈ R^(10×7)
L_A = Huber(A_hat_t, A_t*)
```

`G_psi` 使用 16 个 learned queries 和两层轻量 cross-attention 压缩 576 个 displacement tokens。`D_inv`
只能读取 `B_t^R`，不直接读取当前图像、语言或 `q_t`。

Phase A 的 current-context 对照已经完成：

| 表示输入 | episode-heldout Huber ↓ |
|---|---:|
| `DeltaY_t` | **0.05705** |
| `(Y_t^0,q_t)` | 0.07829 |
| `(DeltaY_t,Y_t^0,q_t)` | 0.06989 |

因此最终教师固定为：

```text
B_t^R = stopgrad(G_psi(DeltaY_t))
```

正式接口不再保留 current/state 输入，也不传入零张量。先前的零输入只属于参数量匹配的对照实验。

### 3.3 `B_t^R` 的含义边界

`B_t^R` 是 **inverse-dynamics-aligned latent transition code**：输入限定它来自 H10 JEPA semantic
displacement，inverse objective 要求它保留可辨识专家动作的信息，低容量瓶颈抑制静态细节。

它不是光流、对象 slot、未来图像或严格因果动作后果；16 个 token 也没有人工指定的固定语义。

## 4. Phase B：Future-Privileged Joint Flow Distillation

### 4.1 当前条件不是 `B_t^P`

冻结 VLM 从当前图像和语言产生 change-query hidden：

```text
H_t^B = FrozenVLM([image, language, Q_1:16^B])
C_t   = P_omega(H_t^B, q_t)
```

`C_t` 只是联合 Flow 的当前条件。它不能称为 `B_t^P`，也不通过单独的
`Huber(B_t^P,B_t^R)` 直接回归 future target。

真正的 `B_t^P` 是部署时与 action 共同积分后得到的 change 输出。

### 4.2 联合 Action–Change Flow

两条流都从独立高斯噪声开始，并共享同一个 `tau`：

```text
A_tau = tau * epsilon_A + (1-tau) * A_t*
B_tau = tau * epsilon_B + (1-tau) * B_t^R

V_A* = epsilon_A - A_t*
V_B* = epsilon_B - B_t^R
```

模型使用 separate action/change parameters，并在 block 内进行双向 joint attention：

```text
(V_A_hat, V_B_hat) = F_theta(A_tau, B_tau, C_t, tau)

L_B = MSE(V_A_hat,V_A*) + lambda_B * MSE(V_B_hat,V_B*)
```

第一版不加入 learned gate、手工 flow-time threshold、tracker、point flow、VQ、reward 或额外对比损失。

部署时：

```text
(A_1,B_1) = (epsilon_A,epsilon_B)
(A_1,B_1) ──ODE conditioned on C_t──► (A_hat_t,B_t^P)
```

蒸馏发生在：训练期 future-privileged `B_t^R` 定义 change-flow 终点，而部署模型仅靠当前条件生成
`B_t^P`。这里对齐的是条件联合分布，不要求手工设置 `B_t^P=B_t^R`。

## 5. A+B 最小验证

### 5.1 为什么先缓存当前 VLM hidden

在修改 π0.5 大模型前，先复用作者 checkpoint 对同一批当前图像和语言产生的冻结 64 个 JEPA-WAM query
hidden，作为信息更丰富的 `H_t^B` 上限条件。该 gate 只验证联合 Flow 机制：

- 如果联合机制在该条件下都无收益，停止，不实现 16 个新 query；
- 如果通过，再把条件接口替换为最终的 16 个 VLM 内生 change queries；
- 64-query gate 不作为最终论文结果或最终模型。

### 5.2 唯一必要对照

训练两个参数量完全相同的模型：

1. `independent`：action/change 分别读取当前 VLM 条件，但两条流互不可见；
2. `joint`：action/change 使用双向 joint attention，共同去噪。

二者使用完全相同的：

- train/validation episodes；
- batch order、flow time 和 Gaussian noise seeds；
- width、depth、heads、optimizer 和训练步数；
- action/change loss 权重。

不使用打乱 transition 或非法反事实输入。

### 5.3 最小运行与判据

```text
320 samples
160 train / 160 validation
500 steps, batch 32
2-layer joint-flow core, width 128
10-step Euler integration
```

episode-level paired bootstrap 同时检查：

```text
action flow loss:       joint < independent
change flow loss:       joint < independent
integrated action MSE:  joint < independent
integrated change MSE:  joint < independent
```

四项 improvement 的 95% CI 下界均需大于 0。任一项失败都原样停止，不通过延长训练或临时增加模块掩盖。

## 6. 通过最小 gate 后的实现顺序

只有 A+B gate 通过才执行：

1. 在冻结 π0.5 VLM prefix 中加入 16 个内生 change queries；
2. 用其输出替换缓存的 64-query upper-bound condition；
3. 将联合 block 接入 π0.5 Action Expert，而不是训练独立 action policy；
4. 验证关闭新模块时严格复现作者 checkpoint；
5. 运行固定 20-episode paired rollout；
6. 出现正向控制信号后才扩大至 LIBERO-Plus L4/L5。

## 7. 可以与不能声称的内容

若最终实验成立，可以表述为：

> We extract an inverse-dynamics-aligned latent displacement from a frozen JEPA and distill this
> future-privileged representation into a current-conditioned joint flow that co-generates continuous actions and
> their predicted latent transitions.

不能声称：

- `B_t^R` 是严格因果动作后果；
- 每个 change token 对应固定物体或空间区域；
- offline flow loss 自动等价于 rollout 成功率；
- 64-query 最小 gate 就是最终 Con1；
- Con1 已经包含 TTA/TTT 或持续学习。
