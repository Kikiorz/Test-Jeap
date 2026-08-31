# 结论先说

你的核心直觉是对的，但需要把一句话改得更严谨：

> 不是“让 \(R\) 学会 action 的模式”，而是让**当前正在生成的动作假设先消除 \(R\) 中的动作歧义**；随后，再让这个已经与当前动作一致的 \(R\) 反过来精化 Action Flow。

我最终建议的 Con1 是：

# Action-Contingent Transition Co-Refinement

# 动作条件转移—动作协同细化

核心信息流：

$$
\boxed{
R_t^{\mathrm{prior}}
\;\xrightarrow[\text{partial action hypothesis}]{A_\tau\rightarrow R}\;
R_{t,\tau}^{\mathrm{act}}
\;\xrightarrow[\text{transition-informed refinement}]{R\rightarrow A}\;
A_\tau^{+}
}
$$

其中没有新建额外语义 latent：

* \(R_t^{\mathrm{prior}}\)：直接使用已经训练好的 JEPA-WAM future-query tokens；
* \(R_{t,\tau}^{\mathrm{act}}\)：原 \(R_t\) 被当前动作假设细化后的工作副本；
* \(A_\tau^{+}\)：读取动作一致转移信息后的 Action Flow hidden。

这不是简单地：

$$
R\rightarrow \text{Action condition}
$$

也不是对称地把两类 token 混在一起，而是一个有明确顺序的：

$$
\boxed{
\text{Action disambiguates transition}
\rightarrow
\text{Transition refines action}
}
$$

---

# 一、问题的本质到底是什么？

## 1. JEPA-WAM 的 \(R\) 是什么？

在 pretrained \(\pi_{0.5}\) 版本中，JEPA-WAM 添加了 64 个 future-query tokens：

$$
R_t=\{r_{t,1},\ldots,r_{t,64}\}.
$$

它们只根据：

$$
o_t,\ell
$$

也就是当前观测和语言任务产生，并通过 alignment head（对齐头）预测冻结 V-JEPA 提供的 joint current–future target（联合当前—未来目标）。这些 token 被组织成 \(8\times 8\) 粗空间网格，但原 \(\pi_{0.5}\) 的感知—动作路径被保留，部署时预测头被移除。([arXiv][1])

因此 \(R_t\) 最稳妥的定义是：

$$
\boxed{
R_t=\text{observation-conditioned transition prior（观测条件转移先验）}
}
$$

它不是：

* 真实未来；
* 某一条候选动作的反事实后果；
* 可直接执行的计划；
* action latent（动作潜变量）。

你的草案已经正确识别了这个缺口：JEPA-WAM 学出了预测转移，但原 Action Flow 在推理时看不到它。

---

## 2. 为什么 \(R_t\) 不能直接作为 Action Expert 的 condition？

最根本的原因不是“token 太多”，而是：

$$
\boxed{
R_t\text{ 没有条件化当前正在生成的动作模式。}
}
$$

从概率角度，可以把 JEPA-WAM 当前预测理解为：

$$
p(Y\mid o_t,\ell)
=
\int
p(Y\mid o_t,\ell,a)\,
p_{\mathcal D}(a\mid o_t,\ell)\,da.
$$

这不是说网络真的显式计算了这个积分，而是说明：训练 target 来自执行专家动作后的未来，但 predictor 输入中没有这条动作。因此，它学到的更接近专家数据分布下的**动作边缘化转移先验**。

当某个状态存在多种合理动作时：

$$
a^{(1)},a^{(2)},\ldots
$$

这些动作可能对应不同转移：

$$
Y^{(1)},Y^{(2)},\ldots
$$

但 \(R_t=f(o_t,\ell)\) 只有一个。

SelfWAM 正面指出，仅根据任务文本和当前观测预测未来，容易学成 generic task progression（通用任务进程），而不是特定动作的后果；它通过让 future queries 在训练时读取 clean demonstrated action（干净专家动作）来提高 action sensitivity（动作敏感度）。([arXiv][2])

这也和你此前自己的诊断一致：JEPA future prediction 明显优于状态保持，但打乱 action 后误差只从 0.449 变成 0.454，说明预测器可以在几乎不利用动作的情况下预测专家未来。

---

## 3. 为什么“完整 future latent → action”也不够？

因为 predictive representation（预测表征）和 control-sufficient representation（控制充分表征）并不相同。

完整 future/transition latent 可能同时描述：

* 静态场景内容；
* 背景和外观；
* 不受机器人控制的变化；
* 遮挡和相机运动；
* 与当前动作阶段无关的局部变化；
* 真正与操作相关的物体—夹爪—目标关系。

MoLA 明确把这个问题称为 perceptual fidelity（感知保真度）与 control relevance（控制相关性）之间的错位：预测未来看起来合理，不意味着其中的信息适合直接解码成动作。([arXiv][3])

JEPA-WAM 自己也没有要求 Action Expert 依赖完整 predicted transition；它使用 dedicated action representation（专用动作表征），并指出直接暴露冗余 future-state information 可能干扰动作生成。([arXiv][1])

因此，真正的问题不是：

$$
\text{如何把 }R\text{ 塞入 Action Expert？}
$$

而是：

$$
\boxed{
\text{如何先把 observation-conditioned }R
\text{ 变成与当前动作假设一致的 }R？
}
$$

---

# 二、调研之后，现有工作分别解决到了哪里？

## 1. Auxiliary WAM：只在训练时使用未来

Fast-WAM 和 JEPA-WAM 证明，未来预测作为联合训练目标本身可以显著改善策略表征，同时部署时可以完全删除未来生成路径，保持快速 action-only inference（仅动作推理）。但代价是预测表征在推理时不再显式参与动作决策。([arXiv][4])

## 2. Action-conditioned future：让世界预测理解动作

SelfWAM 让 clean expert action 条件化 future visual queries，从而使预测更能区分不同动作后果；但这种 action condition 主要存在于训练和可选 world rollout 中，快速部署路径仍然是 action-only。([arXiv][2])

## 3. Future-conditioned action：让动作读取未来

Faster-WAM 证明 inference-time future conditioning（推理期未来条件化）对 OOD 鲁棒性有价值，并通过 SparseMoT 只在少量层/阶段复用 future features，从而改善性能—速度折中。但它的核心仍然主要是：

$$
\text{future representation}\rightarrow\text{action}
$$

的单向使用。([arXiv][5])

DiT4DiT 也从视频去噪过程中提取未来动态特征作为动作条件，但仍属于 future feature 对 action generation 的级联式支持。([arXiv][6])

## 4. Joint-WAM：世界和动作一起生成

Motus、MotuBrain、LingBot-VA、WA-JEPA 等方法表明，world/action 使用 modality-specific experts（模态专用专家）并通过 MoT/MMDiT 式交互共同训练，可以取得很强表现。WA-JEPA 甚至直接联合去噪 future scene tokens 和 future actions。问题是，这类方法通常需要从头或大规模 joint training（联合训练），而不是在已经训练好的 JEPA-WAM checkpoint 上轻量继续训练。([arXiv][7])

## 5. Control-oriented bridge：重新发明中间动作表示

MoLA、DELE-w0.5、LAWA 都说明“视觉未来直接给动作”通常不是最佳接口；它们分别引入 inverse dynamics mixture（逆动力学混合）、compact future latent（紧凑未来潜变量）或 latent action intention（潜在动作意图）。但这会创建新的中间 latent/action space。([arXiv][3])

而你的明确要求是：

$$
\boxed{
\text{不重新发明一个 }C,\text{直接利用已经训练好的 }R.
}
$$

---

# 三、真正的研究空白

综合上述工作，一个比较干净的空白是：

> **能否在不重建 Joint-WAM、不重新定义 latent action、也不进行候选动作 MPC 的前提下，利用 Flow Policy 在推理时天然存在的 partially denoised action hypothesis（部分去噪动作假设），将 JEPA-WAM 的动作边缘化转移先验细化为动作一致的转移表征，再用该表征反向改善同一条 Action Flow？**

我没有在上述已检索工作中发现完全相同的设计。最接近的是：

* SelfWAM：clean expert action \(\rightarrow\) future prediction，但只在训练/rollout 路径使用；
* Faster-WAM：future \(\rightarrow\) action，但主要是单向；
* WA-JEPA：future/action joint denoising，但需要重新构建 Joint-WAM；
* MMDiT：双向多模态交互，但不是 action hypothesis 对预训练 transition prior 的有序细化。

检索不能严格证明“绝对没人做过”，但目前这个边界是清楚的。

---

# 四、最终 Con1：Action-Contingent Transition Co-Refinement

## 一句话定义

> **At each late Action-Flow step, we use the partially resolved action hypothesis to refine JEPA-WAM’s pretrained transition tokens into an action-consistent working transition, and then use that refined transition to correct the remaining action generation.**

中文：

> **在动作流后期，我们利用已经部分形成的动作假设，将 JEPA-WAM 的预训练转移令牌细化为动作一致的工作转移表征，再利用该转移表征精化剩余动作生成。**

完整信息流：

```text
Current observation + language
               │
               ▼
     pretrained JEPA-WAM prefix
        ┌──────────────┐
        │              │
        ▼              ▼
 action prefix P_t   transition prior R_t
        │              │
        │              │
noisy action x_tau     │
        │              │
        ▼              │
pretrained Action Expert early blocks
        │              │
        ▼              │
partial action hidden H_tau
        │              │
        └── A → R ─────┘
               │
               ▼
 action-consistent transition R^A_(t,tau)
               │
               └── R → A ──► refined action hidden
                                  │
                                  ▼
                       pretrained late Action blocks
                                  │
                                  ▼
                           flow velocity v_tau
```

---

# 五、每一个符号的含义

| 符号                 | 含义                                      | 来源                        |
| ------------------ | --------------------------------------- | ------------------------- |
| \(o_t\)            | 当前多视角视觉观测                               | 机器人相机                     |
| \(\ell\)           | 当前语言任务                                  | 任务指令                      |
| \(s_t\)            | 当前 proprioception（本体状态）                 | 机器人关节/末端状态                |
| \(P_t\)            | 原 \(\pi_{0.5}\) Action Expert 的感知语言条件令牌 | JEPA-WAM prefix           |
| \(R_t\)            | 64 个预训练 future-query tokens             | JEPA-WAM prefix           |
| \(Y_{t,t+\delta}\) | V-JEPA joint current–future target      | 仅训练时由真实未来生成               |
| \(x_\tau\)         | 当前部分去噪动作                                | \(\pi_{0.5}\) Action Flow |
| \(H_\tau\)         | Action Expert 中间层动作 hidden              | early action blocks       |
| \(R_{t,\tau}^{A}\) | 当前动作条件下的工作 transition tokens            | Con1 的 A→R 子层             |
| \(H_\tau^{+}\)     | 经过 transition 信息精化的动作 hidden            | Con1 的 R→A 子层             |
| \(v_\tau\)         | Flow Matching velocity（流匹配速度）           | 原 action output head      |

其中：

$$
R_{t,\tau}^{A}
$$

不是新的 latent space。它与 \(R_t\)：

* token 数相同；
* hidden dimension 相同；
* alignment head 相同；
* 空间 token 顺序相同。

它只是当前 flow step 内的一份 working copy（工作副本）。

---

# 六、为什么必须是“有序双向”，而不是 MMDiT 式一次 joint attention？

标准 MMDiT 的原则非常有价值：不同模态使用独立的 Norm、QKV、FFN，但在 attention 中允许双向信息流。它已在大规模 rectified-flow 图像生成中证明优于固定文本条件化。([arXiv][8])

但直接 simultaneous joint attention（同步联合注意力）：

$$
[H_\tau;R_t]
\rightarrow
[H_\tau';R_t']
$$

有一个问题：\(H_\tau'\) 和 \(R_t'\) 都是同时从旧的 \(H_\tau,R_t\) 计算出来的。

因此 Action 在同一个 block 中看到的仍然是旧 \(R_t\)，而不是“已经被当前动作消歧后的 \(R_t'\)”。

我们的语义是有明确顺序的：

$$
\boxed{
H_\tau
\rightarrow
R_{t,\tau}^{A}
\rightarrow
H_\tau^{+}
}
$$

因此采用 **ordered reciprocal block（有序双向块）**，而不是完全对称的 MMDiT block。

我们借用 MMDiT 的：

* modality-specific normalization（模态专用归一化）；
* modality-specific projections（模态专用投影）；
* separate FFNs（独立前馈网络）；
* low-dimensional interaction space（低维交互空间）；

但保留机器人问题所需的有序信息流。

---

# 七、核心子层 1：Action → Transition

## 1. 输入

预训练 transition prior：

$$
R_t\in\mathbb R^{B\times N_R\times d_R},
\qquad N_R=64.
$$

部分动作 hidden：

$$
H_\tau\in\mathbb R^{B\times H\times d_A}.
$$

例如你的配置中：

$$
H=10,\quad d_A=1024,
$$

而 \(d_R\) 由 checkpoint 决定，例如 2048。

---

## 2. 独立归一化

$$
\bar R_t=\operatorname{RMSNorm}_R(R_t)
$$

$$
\bar H_\tau=\operatorname{RMSNorm}_A(H_\tau).
$$

为了避免未来的 JEPA loss 改坏 pretrained Action Flow，对 A→R 路径使用：

$$
\operatorname{sg}(\bar H_\tau),
$$

其中 \(\operatorname{sg}\) 表示 stop-gradient（停止梯度）。

WA-JEPA 在 joint future-action predictor 中也对进入 future scene stream 的 action tokens 使用 stop-gradient，以防 world loss 反向改变 action stream；这是一个已经验证过的稳定设计。([arXiv][9])

---

## 3. 共享交互几何

不创建新的 token，仅将两边投影到同一个 attention width：

$$
Z_R
=
\operatorname{QKNorm}
\left(
W_R\bar R_t
\right)
\in\mathbb R^{B\times64\times d_I}
$$

$$
Z_A
=
\operatorname{QKNorm}
\left(
W_A\operatorname{sg}(\bar H_\tau)
\right)
\in\mathbb R^{B\times H\times d_I}.
$$

建议：

$$
d_I=256,\qquad \text{heads}=8.
$$

\(Z_R,Z_A\) 只是 attention query/key projections，不是新的 \(C\) latent。

---

## 4. Action-conditioned transition update

让 \(R_t\) 中每个 transition token 读取当前动作假设：

$$
M_{R\leftarrow A}
=
\operatorname{MHA}
\left(
Q=Z_R,\,
K=Z_A,\,
V=W_V^A\bar H_\tau
\right).
$$

通过 transition-side output projection：

$$
\Delta R_\tau
=
W_O^R M_{R\leftarrow A}.
$$

然后使用一个标准的 PreNorm Transformer update：

$$
\boxed{
R_{t,\tau}^{A}
=
R_t+
m(\tau)\,
g_R\,
\operatorname{Block}_R
\left(
R_t,\Delta R_\tau
\right)
}
$$

其中：

* \(m(\tau)\)：late-flow mask；
* \(g_R\)：零初始化 gate；
* \(\operatorname{Block}_R\)：一次 output projection + 小 FFN；
* 输出维度仍然是 \(d_R\)。

这不是 residual action policy。它只是把新 block 以 identity-preserving（保持恒等）方式接到 pretrained token space 中。

ReZero 证明零初始化 residual gate 可以让新增深层模块从恒等映射开始并保持良好的梯度传播；DiT 系列也广泛使用 zero-gated modulation 保护预训练生成路径。([arXiv][10])

---

# 八、核心子层 2：Transition → Action

现在不再让 Action 读取原始 \(R_t\)，而读取：

$$
R_{t,\tau}^{A}.
$$

重新投影：

$$
Z_R^{A}
=
\operatorname{QKNorm}
\left(
W_R\operatorname{RMSNorm}_R
(R_{t,\tau}^{A})
\right).
$$

Action queries 使用原来的：

$$
Z_A
=
\operatorname{QKNorm}
(W_A\bar H_\tau).
$$

然后：

$$
M_{A\leftarrow R}
=
\operatorname{MHA}
\left(
Q=Z_A,\,
K=Z_R^{A},\,
V=W_V^R
\operatorname{RMSNorm}_R(R_{t,\tau}^{A})
\right).
$$

生成 action-side update：

$$
\Delta H_\tau
=
W_O^A M_{A\leftarrow R}.
$$

最终：

$$
\boxed{
H_\tau^{+}
=
H_\tau+
m(\tau)\,
g_A\,
\operatorname{Block}_A
(H_\tau,\Delta H_\tau)
}
$$

其中：

* \(g_A\) 零初始化；
* \(\operatorname{Block}_A\) 使用 action-specific FFN；
* 输出仍然位于原 Action Expert hidden space。

随后运行原 Action Expert 剩余 blocks：

$$
H_\tau^{\mathrm{final}}
=
F_{\mathrm{late}}
(H_\tau^{+},P_t,\tau)
$$

再由原 output head 得到：

$$
\boxed{
v_\tau=W_{\mathrm{out}}H_\tau^{\mathrm{final}}.
}
$$

因此 \(R\) 不是在 output projection 前添加一个线性 velocity residual，而是真正参与剩余 Action Expert 的非线性处理。

---

# 九、应该插在哪里？

假设 Action Expert 有 \(L=18\) 层。

第一版建议：

$$
\boxed{
\text{blocks }1:16
\rightarrow
\text{ACTR block}
\rightarrow
\text{blocks }17:18
}
$$

原因是：

* 太早时动作 hidden 仍然受噪声支配；
* 太晚时没有足够网络深度利用 transition；
* 留两层 frozen refinement 是一个保守起点。

你的原 AC-DTR 草案已经采用“16 层形成 draft + 2 层精化”的结构，这一位置选择是合理的；现在只是把单向 deformable readout 改成有监督的 A→R→A co-refinement。

---

# 十、为什么只在 late Action Flow 使用？

按照你的 OpenPI convention：

$$
x_\tau
=
\tau\epsilon+(1-\tau)a^*
$$

其中：

$$
\tau=1\text{ 是纯噪声},\qquad
\tau=0\text{ 是干净动作}.
$$

因此定义：

$$
m(\tau)
=
\mathbf1[\tau\le0.5].
$$

早期：

$$
\tau>0.5
$$

完全使用原 JEPA-WAM。

后期：

$$
\tau\le0.5
$$

才执行：

$$
H_\tau
\rightarrow
R_{t,\tau}^{A}
\rightarrow
H_\tau^{+}.
$$

这符合 SelfWAM 的核心观察：world prediction 只有在获得有意义的 action condition 后，才更可能描述 action-specific consequence；纯噪声动作不能提供这种信息。([arXiv][2])

---

# 十一、训练目标完全复用已有监督

## 1. Transition target

训练时继续使用 JEPA-WAM 原 target：

$$
Y_{t,t+\delta}
=
\operatorname{sg}
E_J
\left(
\operatorname{Stack}_{time}
(o_t,o_{t+\delta})
\right).
$$

不重新定义 semantic flow，也不做 latent subtraction。

更新后的：

$$
R_{t,\tau}^{A}
$$

继续送入原 frozen alignment head：

$$
\hat Y_{t,\tau}
=
G_{\mathrm{align}}
(R_{t,\tau}^{A}).
$$

Transition loss：

$$
\boxed{
\mathcal L_{\mathrm{tr}}
=
\frac{1}{BN}
\sum_{b,n}
\left[
1-
\cos
\left(
\hat Y_{t,\tau,n}^{(b)},
Y_{t,t+\delta,n}^{(b)}
\right)
\right]
}
$$

它约束：

> action information 进入 \(R_t\) 后，\(R_{t,\tau}^{A}\) 仍然必须位于原 JEPA transition space 中。

---

## 2. Action Flow loss

专家动作：

$$
a^*.
$$

噪声：

$$
\epsilon\sim\mathcal N(0,I).
$$

部分动作：

$$
x_\tau=\tau\epsilon+(1-\tau)a^*.
$$

在你的 convention 下 velocity target 为：

$$
v^*=\epsilon-a^*.
$$

Action loss：

$$
\boxed{
\mathcal L_{\mathrm{act}}
=
\mathbb E
\left[
\|v_\tau-v^*\|_2^2
\right].
}
$$

它约束：

> action-conditioned transition 必须真正改善连续动作生成，而不能只提高 JEPA prediction。

总损失：

$$
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{act}}
+
\lambda_{\mathrm{tr}}
\mathcal L_{\mathrm{tr}}
}
$$

第一版沿用 JEPA-WAM 权重：

$$
\lambda_{\mathrm{tr}}=0.1.
$$

不加入：

* InfoNCE；
* cycle consistency；
* energy；
* candidate action；
* segmentation；
* extra latent；
* action residual target。

---

# 十二、推荐两阶段训练

## Stage 1：先验证 Action → Transition

加载你已经训练好的：

$$
\pi_{0.5}+\text{JEPA-WAM}
$$

checkpoint。

冻结：

* VLM；
* future-query tokens；
* alignment head；
* 整个 Action Expert；
* action input/output projection。

只启用：

$$
T_{A\rightarrow R}.
$$

训练：

$$
\mathcal L_{\mathrm{tr}}.
$$

采样：

$$
\tau\in[0,0.5].
$$

目标是先证明：

$$
\boxed{
\mathcal L_{\mathrm{tr}}
(R_{t,\tau}^{A})
<
\mathcal L_{\mathrm{tr}}
(R_t)
}
$$

并且将 action hidden 在同任务、相近状态样本间打乱后：

$$
\boxed{
\mathcal L_{\mathrm{tr}}^{\mathrm{shuffle}}
>
\mathcal L_{\mathrm{tr}}^{\mathrm{correct}}.
}
$$

建议 4k–8k steps。

这一步不成立，整个“双向”故事就停止，不应进入 Stage 2。

---

## Stage 2：闭合 Transition → Action

加载 Stage 1。

启用：

$$
T_{R\rightarrow A}.
$$

训练两个新子层：

$$
T_{A\rightarrow R},
\qquad
T_{R\rightarrow A}.
$$

损失：

$$
\mathcal L_{\mathrm{act}}
+
0.1\mathcal L_{\mathrm{tr}}.
$$

第一轮仍然冻结所有原 checkpoint 参数。

只有新接口确实降低 held-out action loss、但 rollout 增益受限时，才允许给最后两层 Action Expert 增加 rank-8 LoRA。这个 LoRA 是 fallback（后备方案），不是 Con1 必需组件。

---

# 十三、部署过程

每个真实 observation 只运行一次 JEPA-WAM prefix：

$$
(P_t,R_t)
=
F_{\mathrm{prefix}}(o_t,\ell).
$$

缓存：

$$
P_t,\quad R_t.
$$

然后进行原 Action Flow。

```python
def sample_action(observation, language, state, noise):
    action_prefix, transition_prior = prefix_forward(
        observation,
        language,
        return_future_queries=True,
    )

    x = noise

    for tau, next_tau in flow_schedule:
        h = action_input_encoder(x, tau, state)

        h = run_action_blocks(
            h,
            action_prefix,
            start=0,
            end=16,
        )

        if tau <= 0.5:
            # Always restart from the pretrained transition prior.
            transition_work = transition_refiner(
                transition_prior,
                h.detach(),
            )

            h = action_refiner(
                h,
                transition_work,
            )

        h = run_action_blocks(
            h,
            action_prefix,
            start=16,
            end=18,
        )

        velocity = action_output_head(h)
        x = flow_solver_update(x, velocity, tau, next_tau)

    return x
```

每个 flow step 都重新：

$$
R_{t,\tau}^{0}=R_t.
$$

不把上一 flow step 的 \(R^{A}\) 递归传递下去，因为标准 Flow Matching 训练没有监督跨 solver step 的 recurrent transition memory。

部署时不需要：

* future observation；
* V-JEPA teacher；
* alignment head；
* future image generation；
* MPC；
* candidate action search。

额外开销只有 late-flow 中两次长度约 \(H\leftrightarrow64\) 的小型 cross-attention。

Faster-WAM 已证明，future representations 在推理期确实有价值，但最合理的实现是计算一次、稀疏地在动作去噪过程中复用，而不是在每层进行完整 video-action interaction。([arXiv][5])

---

# 十四、为什么它比你之前的 AC-DTR 更强？

原 AC-DTR：

$$
H_\tau
\rightarrow
\text{从 }R\text{ 中读取局部信息}
\rightarrow
H_\tau^{+}.
$$

它本质上仍然是：

$$
R\rightarrow A
$$

的单向接口。你的草案对此描述得很清楚。

现在的 ACTR：

$$
\boxed{
H_\tau
\rightarrow
R_{t,\tau}^{A}
\rightarrow
H_\tau^{+}.
}
$$

它多出的不是“第二个 attention”，而是一个有明确监督的中间步骤：

> **当前动作先解释/消歧转移先验；动作随后读取与自己一致的转移，而不是读取动作无关的通用转移。**

这是一个科学机制变化，而不是容量增加。

---

# 十五、与最危险近邻的边界

| 方法             | 核心机制                                                      | 与我们的不同                                                          |
| -------------- | --------------------------------------------------------- | --------------------------------------------------------------- |
| JEPA-WAM       | Transition prediction 通过共享 backbone 改善动作                  | 我们让 transition 在部署时显式、动作条件化地参与 Action Flow                      |
| SelfWAM        | clean expert action 条件化 future prediction；部署仍 action-only | 我们使用推理时真实存在的 partial action hypothesis，并把 transition 反馈回 action |
| Faster-WAM     | sparse future features 单向条件化动作                            | 我们先执行 \(A\rightarrow R\) 消歧，再执行 \(R\rightarrow A\)              |
| WA-JEPA        | future scene 与 action 从噪声共同生成                             | 我们不重新生成 future latent，只细化 pretrained \(R\)                      |
| MMDiT/MoT      | 模态专用专家 + 双向 joint attention                               | 我们借其参数分离原则，但使用有机器人语义的有序 A→R→A                                   |
| MoLA/DELE/LAWA | 重新构造 latent action 或 inverse-control bridge               | 我们不创建新 latent，直接保留 JEPA-WAM 的 \(R\)                             |
| FWM/V-JEPA2-AC | candidate action consequence + energy/MPC                 | 我们不预测候选动作后果，也不做搜索或能量引导                                          |

上述边界分别由这些工作的方法设计支持。([arXiv][1])

---

# 十六、最小且决定生死的实验

## Gate 1：动作是否真的消歧 \(R\)？

$$
L_{\mathrm{tr}}(R^A)<L_{\mathrm{tr}}(R)
$$

同时：

$$
L_{\mathrm{tr}}(R^A_{\mathrm{shuffled\ action}})
>
L_{\mathrm{tr}}(R^A_{\mathrm{correct\ action}}).
$$

Shuffle 必须在：

* 同任务；
* 相近阶段；
* 尽量相近当前状态；

之间完成，避免模型仅根据任务身份识别错配。

## Gate 2：转移是否真的改善动作？

比较：

$$
\text{JEPA-WAM}
$$

$$
\text{direct }R\rightarrow A
$$

$$
A\rightarrow R\text{ only}
$$

$$
R\rightarrow A\text{ only}
$$

$$
\boxed{A\rightarrow R\rightarrow A}
$$

完整方法必须优于 one-way \(R\rightarrow A\)。

## Gate 3：信息内容是否真实被使用？

将 \(R_t\) batch-shuffle：

$$
R_t^{(b)}\rightarrow R_t^{(b')}.
$$

保持 observation、action noise、flow time 不变。

要求：

$$
L_{\mathrm{act}}\uparrow
$$

且 rollout SR 下降。

## Gate 4：作用是否只来自新增参数？

增加一个等参数量 action-only block：

$$
H_\tau\rightarrow \text{MLP/Attention}\rightarrow H_\tau^{+}
$$

但不读取 \(R\)。

完整方法必须明显优于它。

---

# 十七、最适合的 benchmark

主结果：

$$
\boxed{\text{LIBERO-Plus overall success rate}}
$$

而不是只测试 Layout，因为 Con1 声称的是一般 action–transition consistency。

机制切片重点看：

* Robot Initial State：Action Flow 的 proprioception 和 motor hypothesis 能给 \(R\) 提供额外信息；
* Objects Layout：空间转移先验存在更强歧义；
* Camera Viewpoint：验证方法没有仅记住像素位置。

同时报告标准 LIBERO retention，平均成功率下降最好控制在 1 个百分点以内。

真机最适合：

> 相同语言与物体场景，但改变机械臂初始姿态、接近方向或目标位置。

这会让 observation-conditioned \(R\) 大体相似，但当前动作模式不同，正好检验：

$$
\boxed{
\text{同一 transition prior 能否被不同 action hypothesis 正确消歧。}
}
$$

---

# 十八、为什么它为 Con2 留出了最自然的入口？

Con1 已经建立：

$$
H_\tau
\rightarrow
R_{t,\tau}^{A}
\rightarrow
H_\tau^{+}.
$$

执行动作以后得到真实观测：

$$
o_{t+\delta}^{\mathrm{real}}.
$$

Con2 可以利用 frozen V-JEPA 构造：

$$
Y_{t,t+\delta}^{\mathrm{real}}
=
E_J([o_t,o_{t+\delta}^{\mathrm{real}}]).
$$

然后仅更新：

$$
T_{A\rightarrow R}
$$

或一个非常小的 fast context，使：

$$
G_{\mathrm{align}}
(R_{t,\tau}^{A})
$$

更接近真实 transition。

因为 Con1 已经证明 \(R_{t,\tau}^{A}\) 会直接影响 Action Flow，所以 Con2 的预测更新终于具有一条明确的 control pathway：

$$
\boxed{
\text{realized transition}
\rightarrow
\text{adapt }A\rightarrow R
\rightarrow
R\rightarrow A
\rightarrow
\text{better action}.
}
$$

---

# 最终判断

我认为这版可以作为 ICRA 的 Con1，但有一个不可回避的前提：

$$
\boxed{
A\rightarrow R
\text{ 必须在 held-out 数据上显著改善 transition prediction，}
}
$$

并且：

$$
\boxed{
A\rightarrow R\rightarrow A
\text{ 必须优于单向 }R\rightarrow A.
}
$$

否则“双向”只是一种架构包装，不能作为贡献。

如果这两个结果成立，Con1 的故事会很完整：

> **JEPA-WAM 提供的是动作边缘化的转移先验。我们利用 Flow Policy 在部署时天然存在的部分动作假设，将该先验细化为动作一致的转移表征，再用其精化同一条动作流。由此，在不生成未来视频、不搜索候选动作、不重建 Joint-WAM 的情况下，使 JEPA prediction 第一次显式参与并改善连续动作生成。**

这就是我经过调研后最推荐锁定的 Con1。

[1]: https://arxiv.org/abs/2608.09381 "JEPA-WAM: Learning Vision-Language-Action Policies with Joint-Embedding World Modeling"
[2]: https://arxiv.org/abs/2608.00725 "SelfWAM: A Self-Grounded Unified World Action Model for Fast Robot Control"
[3]: https://arxiv.org/abs/2605.12167?utm_source=chatgpt.com "From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation"
[4]: https://arxiv.org/abs/2603.16666 "[2603.16666] Fast-WAM: Do World Action Models Need Test-time Future Imagination?"
[5]: https://arxiv.org/abs/2608.04404 "[2608.04404] Faster-WAM: Efficient Inference-Time Future Conditioning for Robust World Action Models"
[6]: https://arxiv.org/abs/2603.10448?utm_source=chatgpt.com "DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control"
[7]: https://arxiv.org/abs/2512.13030?utm_source=chatgpt.com "Motus: A Unified Latent Action World Model"
[8]: https://arxiv.org/html/2403.03206v1 "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
[9]: https://arxiv.org/abs/2608.20974 "WA-JEPA: Rethinking the Video JEPA Paradigm for World-Action Modeling in Autonomous Driving"
[10]: https://arxiv.org/abs/2003.04887?utm_source=chatgpt.com "ReZero is All You Need: Fast Convergence at Large Depth"
