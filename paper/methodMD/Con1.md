# Con1：Action-Grounded Change CoFlow 具体实现方案

> 状态：方法与实现合同，尚未开始修改模型代码  
> 分支：`feat/point`  
> 基础模型：作者发布的 π0.5 JEPA-WAM，step `59999`  
> 范围：本文只定义 Con1；不实现 Con2、TTT/TTA 或大规模 rollout

## 1. 一句话定义

我们先从专家动作产生的真实 V-JEPA transition 中学习一个紧凑、动作可辨识的变化变量

```text
B^R ∈ R^(16×16)，
```

再仅根据当前观测和语言预测任务期望的变化先验 `B^P`，最后使用一个双流 Rectified Flow，使动作假设与
变化假设从

```text
(Gaussian action noise, desired-change prior B^P)
```

共同演化为

```text
(expert action chunk, realized action-grounded change B^R)。
```

核心表达为：

```text
(ε_A, B^P) ──Action–Change CoFlow──► (A*, B^R)
```

它不是把 JEPA latent 作为固定 action condition，也不预测像素未来。

## 2. 科学问题与假设

原 JEPA-WAM 可以从当前观测预测 future representation，但原动作路径不读取 future queries。已有实验也
表明：直接 condition、残差、能量引导和候选重排都没有把该 representation 稳定转化为更好的动作。

Con1 检验两个假设：

1. 冻结 V-JEPA 的真实 current–future representation 中，存在可压缩、可由动作识别的变化子空间；
2. 相比固定 `B^P → Action`，让 action 与 change 在同一个 Flow 中双向共同演化，更容易产生控制相关的
   transition–action coupling。

第一个假设不成立时，不训练 CoFlow。第二个假设不成立时，Con1 判定失败，不通过增加模块挽救。

相关方法原则已有实证支持：

- [Delta-JEPA](https://arxiv.org/abs/2606.31232)：通过动作解码提高 JEPA latent displacement 的动作敏感性；
- [WA-JEPA](https://arxiv.org/abs/2608.20974)：联合生成 future latent 与 action，使动作监督塑造 planning
  representation；
- [LAWA](https://arxiv.org/abs/2608.24882)：联合去噪紧凑 latent intention 与可执行动作，并在部署时省去
  future-video generation。

这些工作支持方法原则，但不代替我们在 π0.5 JEPA-WAM 上的验证。

## 3. 数据合同与时间对齐

### 3.1 一条训练样本

动作块长度固定为：

```text
H = 10
```

每条样本必须来自同一专家 episode：

```text
(o_t, language, q_t, A*_[t:t+H-1], o_(t+H)).
```

不得再使用约 `t+31` 的 future target 监督 10-step action chunk。episode 末端使用：

```text
t_future = min(t + H, episode_final_index)
```

但必须保存有效跨度 `H_eff=t_future-t`；首轮训练只使用 `H_eff=H` 的完整片段，避免不同时间尺度混合。

### 3.2 动作表示

π0.5 内部 action tensor 为：

```text
A* ∈ R^(10×32)
```

其中 LIBERO 只有前 7 维为真实控制维，其余为 padding：

- CoFlow 的 `L_action` 完全复用原 π0.5 flow-matching loss 与 32 维张量；
- Phase A 的 inverse decoder 只监督前 7 个物理动作维，避免通过预测 padding 零值虚假降低 loss。

所有动作使用作者 checkpoint 对应的 input transform 与 normalization statistics，不另建归一化规则。

### 3.3 真实 V-JEPA transition

训练期使用冻结 V-JEPA：

```text
Y^R_t = stopgrad(E_J(stack_time(o_t, o_(t+H))))
Y^R_t ∈ R^(576×1408)
```

`Y^R` 只在训练或动作执行完成后可用。部署生成当前动作时禁止读取 future observation。

## 4. 总体结构

```text
Training-only posterior path
o_t, o_(t+H) ── frozen V-JEPA ──► Y^R ── G_ψ ──► B^R
                                                        │
                                                        ▼
                                                inverse decoder
                                                        │
                                                        ▼
                                                       A*

Deployable prior path
o_t, language ── frozen π0.5 prefix ── P_ω ──► B^P

Joint generation
ε_A, B^P ── one Action–Change CoFlow block ──► A*, B^R
```

训练分为 Phase A、B、C，是为了逐项证伪；论文中它们共同定义一个变量和一个联合生成模型，并非三个独立
模块贡献。

## 5. Phase A：学习真实 Action-Grounded Change `B^R`

### 5.1 默认维度

第一版固定：

```text
K   = 16  # change tokens
d_B = 16  # 每个 token 的通道数
B^R ∈ R^(B×16×16)
```

`K=16` 是默认主模型；`K=32` 只在主模型通过后作为容量消融。选择 16 而不是 4，是为了允许多个物体、夹爪、
目标容器及其关系同时被编码；选择 16 而不是默认 32，是为了维持对 `576×1408` teacher representation 的
强压缩并降低 shortcut 风险。

`16×16` 仍不能从数学上保证不会编码动作或任务阶段。“bottleneck 防止复制动作”不能作为论文论证；真正
证据必须来自 episode-heldout 与 same-state intervention。

### 5.2 Controllable Change Encoder `G_ψ`

输入只包含真实 V-JEPA pair tokens，不包含动作、proprioception 或未来标签之外的信息：

```text
Y^R [B,576,1408]
  ↓ LayerNorm
Linear(1408→256)
  ↓
16 learned change queries [16,256]
  ↓ 2 × {Cross-Attention + SwiGLU FFN}
  ↓ Linear(256→16)
  ↓ LayerNorm over d_B
B^R [B,16,16]
```

固定超参数：

```text
query width       256
attention heads   4
query blocks      2
MLP expansion     4
dropout           0
```

不加入 Slot Attention、VQ、tracker、segmentation、point flow、对比损失或显式稀疏正则。

### 5.3 Inverse Decoder `D_inv`

为了堵住已经观察到的 state/trajectory-phase shortcut，v0 明确不输入 `q_t`：

```text
flatten(B^R) [B,256]
  ↓ LayerNorm
Linear(256→256) + SiLU
  ↓ Linear(256→70)
reshape
A_hat_inv [B,10,7]
```

目标为专家动作的 7 个物理维：

```text
L_inv = mean Huber(A_hat_inv, A*_[...,0:7])
```

Phase A 唯一优化目标就是 `L_inv`。`G_ψ` 从未直接看到 `A*`；它只能从真实视觉 transition 中提取能够帮助
恢复动作的信息。

如果不输入 state 的版本完全无法学习，不能直接把 `q_t` 加回来。允许的第二方案是先冻结一个 state-only
baseline，再让 `B^R` 预测剩余动作残差：

```text
A_hat = stopgrad(D_state(q_t)) + D_change(B^R)
```

该方案只有在 v0 失败且 state-only baseline 被严格报告时才启用。

### 5.4 冻结与训练参数

冻结：

- V-JEPA；
- π0.5 / JEPA-WAM；
- 图像与语言编码器；
- 所有 action-policy 参数。

只训练：

- `G_ψ`；
- `D_inv`。

### 5.5 Phase A 阶段验收

训练/验证必须按完整 episode 隔离。首轮可直接复用已有 320 个 H10 专家片段，160 train / 160 validation。

验证至少报告：

1. correct `B^R` 的 inverse action MSE/Huber；
2. within-task shuffled `B^R`；
3. no-change pair `E_J(o_t,o_t)` 经过同一个 `G_ψ`；
4. task-mean action；
5. same-state `base/+x/-x/+y/-y` 干预上的线性 probe 或五路 retrieval。

必须看到：

```text
correct B^R clearly better than shuffled/no-change
```

并且 same-state 指标明显高于随机 `20%`。否则说明 `B^R` 仍主要编码场景或轨迹阶段，停止 Phase B。

## 6. Phase B：预测当前观测下的 Desired Change Prior `B^P`

### 6.1 不复用旧 JEPA-WAM `R`

旧 64 个 future queries 在约 `t+31` 辅助目标下形成，已有 H10 head/query 重训没有迁移到 same-state 控制
干预。因此新 prior predictor 不读取旧 `R`，只读取冻结 π0.5 的 current image-language prefix。

### 6.2 Prior Predictor `P_ω`

为了不让新增 query 反向改变原 prefix token，采用单向 Perceiver/Q-Former-style readout：

```text
frozen π0.5 final prefix hidden [B,N,2048]
  ↓ Linear(2048→256)
  ↓
16 fresh prior queries [16,256]
  ↓ 2 × {Cross-Attention(prefix) + SwiGLU FFN}
  ↓ Linear(256→16)
  ↓ LayerNorm over d_B
B^P [B,16,16]
```

prior queries 可以读取图像和语言 prefix，但 prefix 不读取 prior queries。这样 `P_ω` 在未训练时不会改变
原 π0.5 的动作输出，也不会把动作 token 泄漏给 `B^P`。

### 6.3 训练目标

Phase A 完成后冻结 `G_ψ` 与 `D_inv`：

```text
B^R = stopgrad(G_ψ(Y^R))
B^P = P_ω(prefix(o_t, language))

L_prior = mean Huber(B^P, B^R)
```

只训练 `P_ω`。不联合回传到 VLM，不加入 action loss、InfoNCE 或额外 future decoder。

### 6.4 Phase B 阶段验收

在 episode-heldout 数据上比较：

- learned `B^P`；
- global-mean `B^R`；
- task-conditioned mean `B^R`；
- shuffled task/language 的 `B^P`。

除了 `Huber(B^P,B^R)`，还使用冻结 `D_inv` 报告：

```text
D_inv(B^P) → expert action
```

但该动作解码指标仅用于诊断，不等价于控制成功。只有 learned prior 同时优于 global/task mean，并对
language/task shuffle 敏感，才进入 Phase C。

## 7. Phase C：Action–Change CoFlow

### 7.1 联合 Rectified Flow

沿用 OpenPI 时间方向：

```text
tau = 1  source/noise
tau = 0  target/data
```

动作路径：

```text
ε_A ~ N(0,I)
A_tau = tau * ε_A + (1-tau) * A*
V*_A  = ε_A - A*
```

变化路径：

```text
B_tau = tau * B^P + (1-tau) * B^R
V*_B  = B^P - B^R
```

两条路径共享同一个 `tau`，联合预测：

```text
[V_hat_A, V_hat_B]
  = F_CoFlow(A_tau, B_tau, prefix(o_t,l,q_t), tau)
```

损失只有：

```text
L_CoFlow = L_action + λ_B * L_change

L_action = original π0.5 flow-matching loss
L_change = mean square(V_hat_B - V*_B)
λ_B      = 0.1
```

`tau` 的采样分布完全复用当前 π0.5 训练实现，不单独设计 schedule。

### 7.2 插入位置

当前 `gemma_300m` Action Expert 深度为 18。第一版只插入一个 CoFlow block：

```text
Action Expert blocks 1–14（冻结）
              ↓
       one CoFlow block
              ↓
Action Expert blocks 15–18（冻结）
              ↓
  original action output projection（冻结）
```

实现时使用 `L-4` 计算插入层，并断言当前 checkpoint 的 `L=18`，避免把 14 写死到其他模型变体。

### 7.3 Change Stream

```text
B_tau [B,16,16]
  ↓ Linear(16→256)
H_B [B,16,256]
```

change stream 的 width 为 256，4 heads，每个 head 64 维。

### 7.4 `B → A`：变化帮助动作形成

Action Expert 第 14 层 hidden：

```text
H_A ∈ R^(B×10×1024)
```

计算：

```text
Q_A = W_Q^A RMSNorm(H_A)       # 1024→256
K_B = W_K^B RMSNorm(H_B)       # 256→256
V_B = W_V^B RMSNorm(H_B)       # 256→256

C_A = MHA(Q_A,K_B,V_B)
H'_A = H_A + W_O^(B→A)(C_A)    # 256→1024
```

`W_O^(B→A)` 使用全零初始化。未训练模型在相同 observation 与 flow noise 下必须严格复现作者 checkpoint。

### 7.5 `A → B`：动作假设细化变化

计算：

```text
Q_B = W_Q^B RMSNorm(H_B)                 # 256→256
K_A = W_K^A stopgrad(RMSNorm(H_A))       # 1024→256
V_A = W_V^A stopgrad(RMSNorm(H_A))       # 1024→256

C_B = MHA(Q_B,K_A,V_A)
H'_B = H_B + (1-tau) * W_O^(A→B)(C_B)
H''_B = H'_B + SwiGLU_FFN(RMSNorm(H'_B))
V_hat_B = Linear(256→16)(RMSNorm(H''_B))
```

`W_O^(A→B)` 同样零初始化。`1-tau` 是固定的物理/SNR schedule，不是可学习 gate：

- `tau≈1` 时动作近似纯噪声，不能污染 change stream；
- 随着动作去噪，Action→Change 耦合连续增强；
- 不使用 `tau≤0.5` 的手工硬阈值。

`stopgrad(H_A)` 只阻止 `L_change` 破坏冻结 Action Expert；`L_action` 仍通过 `B→A` 路径训练 CoFlow 参数。

### 7.6 第一版训练范围

冻结：

- π0.5 VLM/prefix；
- 18 层 Action Expert；
- action input/output projections；
- V-JEPA；
- `G_ψ,D_inv,P_ω`。

只训练：

- `B_tau → H_B` input projection；
- 两个方向的 Q/K/V/O projections；
- change-stream RMSNorm 与 FFN；
- change velocity head。

第一版不使用 Action Expert LoRA。只有 offline action loss 明确改善、但 suffix 容量成为可诊断瓶颈时，才
单独讨论后四层 rank-8 LoRA；它不属于 v0。

## 8. 推理过程

部署时不存在 future observation 或 V-JEPA teacher。

每次重规划：

```text
prefix_t = frozen_pi05_prefix(o_t, language, q_t)
B_1      = P_ω(prefix_t)
A_1      ~ N(0,I)
```

然后复用原 π0.5 solver，从 `tau=1` 积分到 `tau=0`：

```text
(A_1,B_1)
   → (A_tau2,B_tau2)
   → ...
   → (A_0,B_0)
```

每一步：

```text
V_A,V_B = CoFlow(A_tau,B_tau,prefix,tau)
A_next  = A_tau - Δtau * V_A
B_next  = B_tau - Δtau * V_B
```

最终只执行：

```text
A_0[...,0:7]
```

`B_0` 是与生成动作共同形成的 action-consistent change hypothesis，不是真实 future，也不是因果保证。

## 9. 最小实验顺序

所有步骤只跑一个 seed 做机制筛选；机制通过后才增加数据和 seeds。

### 9.1 Smoke 0：数据和形状

- 8 条专家样本；
- 检查 H10 配对、episode 边界、action normalization；
- 检查 `Y^R:[B,576,1408]`、`B:[B,16,16]`；
- 确认 inference API 无 future observation 参数。

### 9.2 Smoke A：Action-Grounded Change

```text
data        320 个既有 H10 专家 pair
steps       500
batch       32
optimizer   AdamW
LR          3e-4
```

若 correct/shuffled/no-change 已出现清楚间隔，再训练到 2,000 steps；否则停止。

### 9.3 Smoke B：Change Prior

```text
data        同一 episode split
steps       500
batch       32
optimizer   AdamW
LR          1e-4
```

只有 learned prior 超过 global/task mean 后才延长到 2,000 steps。

### 9.4 Smoke C：Offline CoFlow

先不运行完整 ODE rollout，只验证随机 `tau` 下的联合向量场：

```text
steps       1,000
batch       尽可能 16–32
LR          1e-4
λ_B         0.1
```

同时训练并比较：

1. frozen π0.5 baseline；
2. fixed condition：`B^P` 只单向写入 action，B 不演化；
3. full CoFlow：`A_tau ↔ B_tau` 双向共同演化。

通过条件：

```text
held-out L_action(CoFlow) < held-out L_action(fixed condition)
L_action(shuffled B^P)    > L_action(correct B^P)
L_change(shuffled action) > L_change(correct action)
```

不能只根据训练 loss 判断。

### 9.5 Smoke D：联合积分

只有 Smoke C 通过后才接入原 solver，检查：

- 生成 action MSE；
- 不同 action noise 是否产生不同但匹配的 `B_0`；
- shuffle `B^P` 是否影响动作；
- zero-init 与原 checkpoint 的数值等价性。

Smoke D 通过后，才选择少量 LIBERO/LIBERO-Plus rollout。

## 10. 必要对照与消融

第一轮只保留以下必要项：

| 对照 | 回答的问题 |
|---|---|
| frozen π0.5 JEPA-WAM | 新方法是否优于基础策略 |
| fixed `B^P → Action` | 收益来自 joint evolution，还是普通 condition |
| independent Action/Change flows | 两条 flow 是否真的需要交互 |
| shuffled `B^P` | Action 是否使用了变化先验 |
| shuffled action hidden | Change 是否使用了动作假设 |
| `K=16` vs `K=32` | token 数增加是否真正有益 |

`K=4` 不作为主模型；可在最终论文中作为过强压缩的容量下界。第一轮不做更多 attention、位置编码、稀疏化
或 LoRA 消融。

## 11. 代码修改计划

### 11.1 新增文件

```text
src/openpi/models/action_change_bottleneck.py
    ControllableChangeEncoder
    InverseActionDecoder
    ChangePriorPredictor

src/openpi/models/action_change_coflow.py
    ActionChangeCoFlowBlock
    action/change interpolants

scripts/train_action_change_bottleneck.py
    Phase A 训练和 same-state 诊断

scripts/train_change_prior.py
    Phase B 训练和 prior baselines

scripts/run_action_change_coflow_gate.py
    Phase C/D 最小对照
```

### 11.2 修改文件

`src/openpi/models/pi0_config.py` 增加默认关闭的配置：

```text
use_action_change_coflow: bool = False
change_num_tokens: int = 16
change_token_dim: int = 16
change_hidden_dim: int = 256
change_num_heads: int = 4
change_injection_from_end: int = 4
change_loss_weight: float = 0.1
```

`src/openpi/models/gemma.py`：

- 为现有 stacked/scan 参数增加等价的 layer-range forward；
- 支持 `blocks 1:L-4 → hook → blocks L-3:L`；
- 不改变任何原 block 的参数名称、shape、attention mask 或 checkpoint 路径。

`src/openpi/models/pi0.py`：

- prefix forward 输出只读 hidden/KV；
- suffix 在指定层调用一个可选 CoFlow hook；
- 训练时同时返回 `V_hat_A,V_hat_B`；
- 推理时联合更新 `A_tau,B_tau`；
- `use_action_change_coflow=False` 时执行路径必须与上游完全一致。

`src/openpi/training/config.py`：

- 增加独立 Phase A/B/C 配置；
- 从 step `59999` warm-start；
- 输出到全新 checkpoint 目录，不覆盖作者 checkpoint。

## 12. 结构正确性测试

必须新增以下单元测试：

1. `B^R/B^P/B_tau/V_B` shape 全部正确；
2. `tau=1` 得到 `(A_tau,B_tau)=(ε_A,B^P)`；
3. `tau=0` 得到 `(A_tau,B_tau)=(A*,B^R)`；
4. `W_O^(B→A)=0` 时动作输出严格等于原 checkpoint；
5. `tau=1` 时改变 action hidden 不改变 `A→B` 更新；
6. `tau<1` 时改变 action hidden 会改变 change output；
7. `L_change` 对 Action Expert 的梯度为零；
8. `L_action` 对 `B→A` 参数梯度非零；
9. baseline layer-range forward 与原 18-layer scan 数值一致；
10. 训练和推理路径均不会意外读取 future observation。

## 13. 如何区分代码错误与方法失败

以下属于代码错误，必须修复后重跑：

- current/future 跨 episode；
- `H_eff≠10` 仍进入首轮训练；
- 动作 normalization 与作者 policy 不一致；
- `B^P/B^R` 使用不同 LayerNorm 或 token 顺序；
- zero-init 无法复现 baseline；
- `L_action` 到 `B→A` 没有梯度；
- `L_change` 意外更新 Action Expert；
- 训练使用 future observation，而推理接口无法删除它；
- layer-range forward 改变原 checkpoint 数值。

以下在上述检查全部通过后，属于方法失败：

- `B^R` 与 shuffled/no-change 无差异；
- `B^P` 不超过 task mean；
- correct 与 shuffled `B^P` 的 action loss 相同；
- correct 与 shuffled action 的 change loss 相同；
- CoFlow 不超过 matched fixed condition；
- offline loss 改善但联合积分动作变差；
- 提升只在 `K=32` 和大规模解冻后出现。

## 14. 可以与不能声称的内容

如果全部核心实验通过，可以声称：

> 我们在冻结 V-JEPA 中学习动作可辨识的紧凑变化空间，并提出 Action–Change CoFlow，使期望变化与连续
> 动作在 π0.5 生成过程中共同演化，而不是通过固定条件单向连接。

当前不能声称：

- `B^P` 是真实未来；
- `B^R` 是严格因果变化；
- 16 个 token 对应 16 个固定物体或空间区域；
- action reconstruction 自动保证更高任务成功率；
- Con1 已经具有测试时持续学习能力；
- offline action loss 改善等价于 LIBERO-Plus 成功率改善。

## 15. 已锁定的第一版决定

1. 使用 `K=16,d_B=16`；`K=32` 仅作后续容量消融；
2. 不使用旧 JEPA-WAM `R` 作为新变化变量；
3. Phase A inverse decoder 默认不输入 proprioception；
4. `G_ψ` 不读取动作，避免直接 action leakage；
5. Phase A/B 仅使用专家数据，同状态扰动数据只用于诊断；
6. 只使用一个 CoFlow block，插在 Action Expert 倒数四层之前；
7. 基础 π0.5、VLM、V-JEPA 和全部 Action Expert 在 v0 中冻结；
8. 不使用 tracker、point flow、VQ、MPC、energy、progress、手工硬 gate 或额外奖励；
9. 在 offline 双向耦合证据出现前，不接完整 solver，不跑大规模 rollout；
10. Con2/TTT 不属于本文件范围，必须等待 Con1 被控制实验支持后再设计。
