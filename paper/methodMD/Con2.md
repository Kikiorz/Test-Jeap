# Con2：Achieved-Change Directional Adaptation

## 1. 核心定义

Con1 在执行前根据当前观测联合生成动作与预测变化：

\[
(o_t,\ell)\longrightarrow (A_t^P,B_t^P).
\]

这里的 \(B_t^P\) 只是模型在执行前生成的 predicted change；此时没有真实 future，也没有
\(B_t^{ach}\)。机器人执行完整动作块并真正观察到 \(o_{t+10}\) 后，才可以用 Con1 冻结的 Change
坐标系编码已经发生的变化：

\[
(o_t,o_{t+10})\longrightarrow B_t^{ach}.
\]

同时机器人知道自己实际发送的控制命令 \(A_t^{exec}\)。因此每个已经完成的动作块自然产生一条不需要
reward 或人工标签的 action–effect 样本：

\[
\boxed{(o_t,\ell,B_t^{ach},A_t^{exec})}.
\]

Con2 把 achieved Change 固定为条件，让同一个 Action–Change MMDiT 反向恢复实际动作，并且只更新一个
很小的、严格从 Change 指向 Action 的方向性低秩接口：

\[
\boxed{B_t^{ach}\rightarrow A_t^{exec}\rightarrow
\text{adapt }(B\rightarrow A)}.
\]

完整闭环是：

```text
Predict → Act → Observe → Encode Achieved Change → Inverse Adapt → Predict Again
```

Con2 不训练第二个 inverse policy，不更新 V-JEPA2、JEPA-WAM future predictor 或 Change tokenizer，也不把
错误动作称为正确示范。它校准的是：在当前部署动力学下，Action stream 应该如何读取真实观察到的 Change。

---

## 2. Con1 与 Con2 是否一起训练

结论是：**共享同一个网络和 Change 坐标系，但主体参数不能同时训练。优化必须按顺序进行。**

### Phase 1：训练 Con1 Stage 1

使用专家视频训练 Change tokenizer 与仅训练期存在的 direct inverse decoder：

\[
(o_t,o_{t+10})\rightarrow E_J\rightarrow \Delta Y_t
\rightarrow G_\psi\rightarrow B_t^R\rightarrow D_{inv}\rightarrow A_t^*.
\]

完成后冻结：

- V-JEPA2 encoder \(E_J\)；
- 无参数空间算子 \(S\)；
- Change tokenizer \(G_\psi\)；
- Change 标准化统计量 \((\mu_B,\sigma_B)\)。

这一步固定了以后所有 \(B\) 的坐标系。若在线阶段还更新 \(G_\psi\)，同一种物理变化会随时间获得不同编码，
inverse supervision 的输入语义就会漂移。

### Phase 2：只训练 Con1 的 joint generation

按照 Con1 的定义，只使用同步的 Action–Change joint flow：

\[
\tau_A=\tau_B=\tau,
\]

\[
\mathcal L_{Con1}=\mathcal L_A+\lambda_B\mathcal L_B.
\]

得到完整 Con1 参数 \(\theta_{Con1}\) 后，将其全部冻结。Con1 阶段不混入 clean-Change inverse loss；否则
Con1 的提升可能来自额外的 \(B^R\rightarrow A^*\) 特权监督，而不是联合生成机制本身。

### Phase 3：离线准备 Con2 adapter

在冻结的 \(\theta_{Con1}\) 上加入 rank-4 directional adapter \(\phi\)。adapter 以“函数为零”的方式初始化，
所以刚加入时不改变已经训练好的 Con1。随后仅优化 \(\phi\)，用同一专家数据交替训练：

- 70% synchronous joint mode，保持 adapter 对 Con1 正常生成路径的兼容；
- 30% clean-Change inverse mode，学习部署时需要的 \(B\rightarrow A\) 条件逆映射。

训练结束得到离线初始化 \(\phi_0\)。这一步属于 Con2 的 offline preparation，不回写 Con1 主体。

### Phase 4：部署时在线更新

部署开始时载入 \((\theta_{Con1},\phi_0)\)。每个 episode 内只更新 \(\phi\)，episode 结束后丢弃 fast-weight
偏移并恢复 \(\phi_0\)。

因此准确顺序是：

```text
Con1 Stage 1 → freeze Change coordinates
Con1 Stage 2 joint flow → freeze complete Con1
Con2 offline adapter preparation → obtain φ0
Con2 deployment → update only episode-local φ
```

---

## 3. Achieved Change 如何得到

### 3.1 记录真实执行窗口

Con1 输出 \(A_t^P\) 后，还会经过 action denormalization、安全裁剪和环境控制接口。Con2 保存的标签必须是
真正发送到机器人并覆盖 \([t,t+9]\) 的 10 个动作：

\[
A_t^{exec}=[a_t^{exec},\ldots,a_{t+9}^{exec}].
\]

不能直接保存 \(A_t^P\)，因为后处理后两者可能不同。完整执行 H10 后，环境才产生真实观测
\(o_{t+10}\)。如果 episode 提前结束或只执行了部分 chunk，该 transition 不进入第一版在线更新。

### 3.2 复用完全相同的 Change 坐标系

Con2 严格复用 Con1 Stage 1 的冻结路径：

\[
Z_t^{ach}=\operatorname{sg}E_J([o_t,o_{t+10}]),
\qquad
Z_t^0=\operatorname{sg}E_J([o_t,o_t]).
\]

使用相同的逐 token normalization 与无参数 \(3\times3\) 空间池化：

\[
S(Z)=\operatorname{L2Norm}\left[
\operatorname{AvgPool}_{3\times3}(\operatorname{L2Norm}(Z))
\right].
\]

构造：

\[
\Delta Y_t^{ach}=S(Z_t^{ach})-S(Z_t^0)
\in\mathbb R^{8\times8\times1408},
\]

\[
B_{t,raw}^{ach}=G_\psi(\Delta Y_t^{ach}),
\]

\[
\boxed{
B_t^{ach}=\frac{B_{t,raw}^{ach}-\mu_B}{\sigma_B+10^{-6}}
\in\mathbb R^{16\times128}
}.
\]

这里没有固定 JL 投影，也没有第二套 achieved encoder。\(E_J\)、预处理、空间池化、\(G_\psi\) 和标准化
统计必须与 Stage 1 逐项相同。

为了减少在线等待，可以在动作块执行期间预先计算并缓存：

\[
Z_t^0=E_J([o_t,o_t]).
\]

获得 \(o_{t+10}\) 后只需再计算真实 pair 分支。该优化不改变算法语义。

---

## 4. 同一个 MMDiT 的两种时间接口

Con1 实现时就保留模态专属时间接口：

\[
F_{\theta}
(A_{\tau_A},B_{\tau_B},C_t,C_t^J,\tau_A,\tau_B).
\]

Action 和 Change 使用各自的 timestep embedding 与 AdaRMSNorm conditioning，但仍在同一个 MMDiT joint
attention 中交换信息。Con1 虽然始终传入 \(\tau_A=\tau_B\)，仍不应把代码接口写死成一个 timestep，否则
Con2 无法在不改网络语义的情况下固定 clean Change。

### 4.1 Con1 的 synchronous joint mode

\[
A_\tau=\tau\epsilon_A+(1-\tau)A_t^*,
\]

\[
B_\tau=\tau\epsilon_B+(1-\tau)B_t^R,
\]

\[
V_A^*=\epsilon_A-A_t^*,\qquad
V_B^*=\epsilon_B-B_t^R.
\]

此时：

\[
\tau_A=\tau_B=\tau.
\]

### 4.2 Con2 的 clean-Change inverse mode

把真实 Change endpoint 固定在数据端：

\[
\boxed{B_0=B_t^R\text{ 或 }B_t^{ach},\qquad \tau_B=0}.
\]

只对动作加噪：

\[
A_{\tau_A}=\tau_A\epsilon_A+(1-\tau_A)A,
\]

并只预测 action velocity：

\[
\widehat V_A^{inv}
=F_{\theta,\phi}^{A}
(A_{\tau_A},B_0,C_t,C_t^J,\tau_A,0).
\]

该模式不构造 Change interpolant、不执行 Change ODE、不预测 Change velocity，也不计算 \(\mathcal L_B\)。
它不是新的 inverse network，而是同一个 CoFlow 在一个模态已知时的条件边界模式。

---

## 5. 严格方向性的 Change→Action adapter

### 5.1 为什么不能直接对公共 Change K/V 做 LoRA

在普通 fused MMDiT 实现中，Change 的 \(K_B,V_B\) 同时被 Action queries 和 Change queries 使用。若直接
修改公共的 \(W_{K_B},W_{V_B}\)，在线更新会同时改变：

\[
B\rightarrow A
\quad\text{和}\quad
B\rightarrow B.
\]

这会改写 Change self-dynamics，不符合“只校准 Change 如何被 Action 读取”的目标。因此 directional
adapter 不能通过给第三个 expert 普通挂 LoRA 来近似实现；它需要显式处理 Action-query rows。

### 5.2 Block-specific attention residual

对每个 MMDiT block 13–18，先照常计算冻结的 base Q/K/V。设 Action 对 Change 的 base logits 为：

\[
S_{AB}=Q_AK_B^\top/\sqrt{d_h}.
\]

rank-4 adapter 只从 Change hidden 产生增量：

\[
\Delta K_B=\operatorname{LoRA}_{K,l}(H_B),
\qquad
\Delta V_B=\operatorname{LoRA}_{V,l}(H_B).
\]

只对 Action-query、Change-key 这一块修改 logits：

\[
\boxed{
S'_{AB}=S_{AB}+Q_A\Delta K_B^\top/\sqrt{d_h}
}.
\]

同时，只有 Action query rows 在聚合 Change values 时使用：

\[
V_B^A=V_B+\Delta V_B.
\]

于是 Action 输出为：

\[
O_A=\operatorname{Attn}
\left(Q_A,[K_C,K_A,K_B+\Delta K_B],
[V_C,V_A,V_B+\Delta V_B]\right),
\]

而 Change rows 完全保持 base Con1：

\[
O_B=\operatorname{Attn}
\left(Q_B,[K_C,K_A,K_B],[V_C,V_A,V_B]\right).
\]

因此 adapter 的直接作用严格局限于：

\[
\boxed{\text{Action queries 如何读取 Change keys/values}}.
\]

实现上需要分别计算或覆盖联合注意力矩阵的 A rows 与 B rows，再拼回输出；不能使用一组已经被 LoRA 改写的
公共 K/V 完成所有 rows。

### 5.3 零函数初始化

每个 rank-4 adapter 采用一侧小随机、另一侧零初始化：

```text
down projection   small random
up projection     zeros
```

这样初始 \(\Delta K_B=\Delta V_B=0\)，所以加入 adapter 时严格复现已经训练好的 Con1，同时 zero side 能在
第一次更新获得非零梯度。不能把 low-rank 两侧都初始化为零，否则两侧梯度也同时为零。

---

## 6. Con2 的离线准备

固定全部 \(\theta_{Con1}\)，只训练 \(\phi\)。每个 batch 选择一种模式。

### 6.1 Joint compatibility mode

以概率 0.7 使用 Con1 原始同步 joint flow，计算：

\[
\mathcal L_{joint}=\mathcal L_A+\lambda_B\mathcal L_B.
\]

梯度只更新 \(\phi\)。这个模式防止 adapter 只适合 clean \(B\) 而破坏正常从噪声联合生成
\((A^P,B^P)\) 的路径。

### 6.2 Privileged inverse mode

以概率 0.3 使用专家 \(B_t^R\) 作为 clean condition，动作噪声时间采用：

\[
\tau_A\sim
0.5\,\delta(1)+0.5\,U(0.8,1.0),
\qquad \tau_B=0.
\]

其中 \(\tau_A=1\) 的样本完全不向模型输入动作标签；高噪声区间样本则提供邻近边界的稳定训练。inverse loss
只计算 LIBERO 的 7 个物理动作维度：

\[
\mathcal L_{inv}
=\operatorname{mean}
\left(M_{valid}\odot
\|\widehat V_A^{inv}-(\epsilon_A-A_t^*)\|_2^2\right).
\]

离线目标为：

\[
\mathcal L_{Con2-offline}
=0.7\,\mathbb E\mathcal L_{joint}
+0.3\,\lambda_{inv}\mathbb E\mathcal L_{inv},
\]

但无论哪种模式，optimizer tree 都只包含 directional adapter。得到的 \(\phi_0\) 是一个已经理解 Con1 Change
坐标、又不破坏 joint generation 的起点。

---

## 7. 部署时在线更新

### 7.1 动作必须回到 policy 训练坐标

环境记录的 \(A_t^{exec}\in\mathbb R^{10\times7}\) 通常处于物理控制尺度。构造 flow loss 前，必须使用
Con1/π0.5 相同的训练集 normalization，把它转换回 policy action space，并按 checkpoint interface 补齐到
\(10\times32\)。不能把物理单位直接与标准高斯噪声插值。

### 7.2 首版使用纯噪声 action condition

为彻底避免 action interpolant 泄漏标签，第一版在线更新固定：

\[
\boxed{\tau_A=1,\qquad\tau_B=0}.
\]

对同一 transition 采样少量独立噪声，例如 \(K=4\)：

\[
A_1^{(k)}=\epsilon_A^{(k)},
\]

\[
V_{A,target}^{(k)}=\epsilon_A^{(k)}-A_t^{exec}.
\]

这样模型输入端看不到 \(A_t^{exec}\)，动作只作为 velocity label 出现。在线 inverse loss 为：

\[
\mathcal L_{ITA}
=\frac1K\sum_{k=1}^K
\operatorname{mean}\left(
M_{valid}\odot
\|\widehat V_A^{(k)}-V_{A,target}^{(k)}\|_2^2
\right).
\]

为限制 fast weights 漂移，加入以离线起点为中心的 proximal term：

\[
\mathcal L_{online}
=\mathcal L_{ITA}+\lambda_{prox}\|\phi-\phi_0\|_2^2.
\]

每个完整 H10 transition 只执行一次 clipped Adam 更新：

\[
\phi_{t+1}=\phi_t-\eta\nabla_{\phi_t}\mathcal L_{online}.
\]

若纯噪声版本学习信号不足，第二个预注册选择才是 \(\tau_A\sim U(0.8,1)\)，而不是降低到 0.5。低于 0.8
会让 interpolant 已含较多真实动作，使 support loss 下降很难证明模型利用了 achieved Change。

### 7.3 更新作用于下一动作块

第 \(t\) 个 chunk 完成后才能获得 \(B_t^{ach}\)，所以更新不可能修正已经执行的动作。更新完成后，机器人处于
新状态 \(o_{t+10}\)，重新计算新状态的 ordinary prefix 与 future context，再用正常 joint mode生成下一块：

\[
(A_{t+10}^P,B_{t+10}^P)
=\operatorname{CoFlow}(o_{t+10},\ell;\theta_{Con1},\phi_{t+1}).
\]

episode 内持续使用 \(\phi_{t+1}\)，episode 结束恢复 \(\phi_0\)：

```text
episode start     φ ← φ0
after each H10    one achieved-change inverse update
episode end       discard φ − φ0
```

第一版不使用 replay buffer、跨 episode memory、EMA teacher 或长期参数写回。

---

## 8. 为什么不优化 predicted-change error

执行前生成的 \(B_t^P\) 和执行后观测到的 \(B_t^{ach}\) 可以计算诊断误差：

\[
e_B=\|B_t^P-B_t^{ach}\|.
\]

但它不进入第一版在线 loss。最小化该误差主要是在修正 future predictor，不能直接保证 Action stream 更懂得
“怎样的动作对应这种真实变化”。它还可能把一次偶然外界扰动写回 JEPA future prediction。

Con2 的核心监督始终是：

\[
\boxed{B_t^{ach}\rightarrow A_t^{exec}},
\]

而 \(B_t^P-B_t^{ach}\) 只用于分析预测与实际是否偏离。

---

## 9. 为什么这种自监督可能有用，以及边界

即使某个动作没有完成任务，下面的事实配对仍然成立：

\[
A_t^{exec}\longleftrightarrow B_t^{ach}.
\]

Con2 学到的不是“这个动作值得重复”，而是“在这个状态和部署动力学下，这个动作事实上产生了这种变化”。
当故障来自持续性的 action-effect shift，例如摩擦、payload、低层控制偏差或 embodiment 差异时，过去 chunk
形成的局部校准可能改善下一 chunk 的生成。

但没有 reward 时，Con2 不知道 achieved Change 是否朝任务目标前进。因此它不能保证修复：

- 错误的语言理解；
- 错误的目标物选择；
- 错误的高层计划；
- 一次性、不可重复的外界扰动；
- JEPA future context 本身方向错误。

这也是为什么在线只更新小型、episode-local、方向性的 adapter，而不更新整个策略或 future predictor。

---

## 10. 三种 Change 变量不能混淆

### \(B_t^R\)：privileged expert Change

训练阶段由专家轨迹的真实 \((o_t,o_{t+10})\) 构造，是 Con1 Change flow 的数据 endpoint，也是 Con2
offline inverse preparation 的 clean condition。

### \(B_t^P\)：pre-execution predicted Change

Con1 在动作执行前与 Action 一起从噪声生成的 endpoint。此时不存在真实 achieved future。

### \(B_t^{ach}\)：post-execution achieved Change

动作真正执行并观察到 \(o_{t+10}\) 后才编码得到，是 Con2 在线 inverse adaptation 的 clean condition。

```text
B^R    expert-future privileged endpoint
B^P    generated prediction before execution
B^ach  encoded outcome after execution
```

三者使用同一个 learned Change coordinate system，但来源和可用时刻不同。

---

## 11. 完整算法

### 离线阶段

```text
1. Train Stage-1 Gψ and Dinv from expert V-JEPA2 displacement.
2. Freeze Gψ and discard Dinv.
3. Train Con1 only with synchronous Action–Change joint flow.
4. Freeze the complete Con1 model θCon1.
5. Add zero-function rank-4 directional adapters φ.
6. Train only φ with 70% joint mode + 30% clean-Change inverse mode.
7. Save φ0.
```

### 每个部署 episode

```text
φ ← φ0

while task not finished:
    use current o_t and language to jointly generate A_t^P, B_t^P
    execute and record the actual normalized command A_t^exec
    observe o_(t+10)
    encode B_t^ach with frozen E_J, S, Gψ, μB, σB
    clamp B=B_t^ach, τ_B=0; use K pure-noise action samples at τ_A=1
    update only directional B→A adapter φ once
    continue from o_(t+10)
```

---

## 12. 学术定位

Con2 不是强化学习，因为它没有 reward、value、advantage 或 policy-gradient return。它也不是普通 TTA：更新
目标不是图像重建或 entropy minimization，而是由真实执行自然产生的 inverse action–change supervision。

最准确的表述是：

> **After training and freezing the predictive Action–Change CoFlow, we prepare a directional Change-to-Action
> adapter under joint and clean-Change inverse modes. At deployment, post-execution observations are encoded in the
> same frozen Change coordinates and provide reward-free supervision for episode-local adaptation of only this
> directional coupling.**

中文概括为：

> **Con1 在执行前让预测未来参与动作—变化联合生成；Con2 在执行后把真实 achieved Change 反向映射到实际
> 动作，只在线校准 Change→Action 的读取关系。**

可以声称：

- Con1 与 Con2 共享同一个 MMDiT 和 Change 坐标系；
- Con2 的主体训练严格发生在 Con1 冻结之后；
- 真实后续观测提供无 reward 的 inverse-dynamics supervision；
- adapter 只改变 Action queries 读取 Change K/V 的块；
- 在线更新不改变 JEPA-WAM 对未来的预测。

不能声称：

- \(B_t^{ach}\) 在执行前已经存在；
- 错误动作被后见地变成了正确示范；
- inverse loss 下降必然提升任务成功率；
- Con2 能修正任务理解或高层规划错误；
- 普通公共 K/V LoRA 已实现严格单向适配；
- Con1 与 clean-future inverse supervision 一起训练仍是公平的 Con1。
