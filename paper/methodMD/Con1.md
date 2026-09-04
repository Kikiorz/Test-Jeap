# Con1：V-JEPA2 Action–Change CoFlow

## 1. 核心思想

π0.5 JEPA-WAM 已经具备一种很有价值、但尚未真正进入控制路径的能力：它在当前图像和任务语言上放置
64 个 future-query tokens，并用 V-JEPA2 的当前—未来视频表征监督这些 tokens。也就是说，VLM 已经被训练
去预测“接下来场景可能如何演化”。

问题在于，原模型默认阻断 Action Expert 对 future queries 的访问。因此，JEPA 预测只服务于辅助表征损失，
并不参与连续动作的生成：

```text
current observation + language
              │
              ▼
      JEPA-WAM future queries ──► auxiliary JEPA loss
              │
              X
              │
              ▼
         Action Expert
```

这不是简单地把 future tokens 拼接到动作 token 上就能解决的问题。完整 JEPA latent 同时编码场景外观、物体
身份、空间布局、任务语义以及时间变化，其中只有一部分与当前控制有关。若直接把它作为普通 condition，动作
模型仍需在高维混合信息中自行寻找有效变化，而且 JEPA latent 与连续 action flow 之间没有明确的学习接口。

Con1 因而引入一个中间生成变量：**Change**。

- 训练期利用真实未来观测构造一个紧凑、带空间结构、可辨识专家动作的 change endpoint
  \(B_t^R\)；
- 部署期保留 JEPA-WAM 从当前观测预测的 future tokens \(R_t^J\)，将其转化为 Change stream 的预测未来
  上下文；
- Action 与 Change 不采用单向条件注入，而是在 MMDiT 中作为两个正在生成的连续模态共同去噪、双向推理。

完整信息链为：

```text
VLM predicted future R_t^J
              ↓
     predicted Change stream
              ↕
       continuous Action stream
              ↓
          robot action
```

因此，本方法的核心不是“让 JEPA 直接预测动作”，而是：

> **让 JEPA 预测未来变化，让 Action Expert 在生成动作的同时生成并解释该变化。**

---

## 2. 为什么需要 Change 这一中间变量

JEPA 更擅长在表征空间描述未来状态或状态变化，而机器人策略需要输出低层连续动作。这两者并不天然处在同一
语义层级：

```text
JEPA latent：场景接下来可能发生什么
Action：机器人关节或末端执行器现在应该怎样运动
```

直接建立 \(R_t^J\rightarrow A_t\) 会把两个问题混在一起：

1. 从 future latent 中分离真正的时间变化；
2. 判断哪些变化与当前任务和动作有关；
3. 将这种变化转换为连续 motor command。

Con1 将它们拆成连续而统一的两个阶段：

1. **Stage 1：定义动作相关的变化坐标系。** 真实当前—未来观测经过冻结 V-JEPA2 后，先减去同帧
   no-change reference，再通过 inverse-dynamics objective 压缩为 \(B_t^R\)。
2. **Stage 2：学习未来与动作的联合生成。** JEPA-WAM 预测的 future context 进入 Change stream，
   Action 与 Change 在同一个 flow model 中共同生成。

这里的 \(B_t^R\) 不是额外 world model 的输出，也不是人工定义的光流。它是连接“预测性视频表征”与“连续
动作生成”的控制接口。

---

## 3. 总体结构

设当前时刻为 \(t\)，策略输出长度为 \(H=10\) 的 action chunk：

\[
A_t^*=\left[a_t,a_{t+1},\ldots,a_{t+9}\right].
\]

训练期使用同一条专家轨迹中的真实未来观测 \(o_{t+10}\)，但该观测只用于构造 Change teacher：

```text
                         Stage 1：训练期特权变化目标

 (o_t, o_t+10) ── frozen V-JEPA2 ──► semantic displacement ΔY_t
                                              │
                                  spatial Change Resampler
                                              │
                                              ▼
                                      B_t^R : 4×4×128


                         Stage 2：部署可用的联合生成

 current images + language ── π0.5 JEPA-WAM ──► ordinary prefix C_t
                    │                                  │
                    └──► future queries R_t^J          │
                                  │                    │
                     frozen alignment + pooling       │
                                  │                    │
                                  ▼                    │
                       predicted future context C_t^J │
                                  │                    │
                                  ▼                    ▼
                         noisy Change B_τ      noisy Action A_τ
                                  └──────────┬─────────┘
                                             ▼
                              Action–Change MMDiT blocks
                                     Action ↔ Change
                                  ┌──────────┴──────────┐
                                  ▼                     ▼
                              A_hat_t                B_t^P
```

训练完成后，真实未来 \(o_{t+10}\)、V-JEPA2 target encoder、Stage-1 inverse decoder 和 \(B_t^R\) 均从
部署路径消失。部署只依赖当前观测、语言以及两组高斯噪声。

---

## 4. 时间尺度：为什么主目标使用 H10

Con1 的 Change 表示应尽量对应当前 action chunk 所造成的视觉变化。因此，Stage 1 的主时间跨度定义为：

\[
(o_t,o_{t+10})\quad\longleftrightarrow\quad A^*_{t:t+9}.
\]

这样 \(B_t^R\) 描述的变化窗口与策略一次生成的 10 步动作窗口一致。

作者 JEPA-WAM checkpoint 的 future queries 则由 `future_offset=31` 的 V-JEPA target 训练。这里的 31 表示
当前帧到第 31 个后续帧之间的视觉跨度，并不表示 π0.5 生成 32 个动作再只使用前 10 个。这个较长跨度可能
更适合表示任务级未来意图，但它也包含当前 10 步动作之后的场景演化。

因此，两个变量承担不同角色：

- \(R_t^J\)：由当前观测预测的、偏长时的 action-agnostic future context；
- \(B_t^R\)：由真实 H10 视频变化构造的、与当前 action chunk 对齐的训练终点。

Con1 不假设二者逐 token 相等。MMDiT 学习的是“较长时的未来预测如何在当前 action chunk 中转化为可生成的
短时变化和动作”，而不是把 H31 future latent 强行回归成 H10 change code。

---

## 5. Stage 1：构造动作可辨识的空间 Change tokens

### 5.1 冻结 V-JEPA2 视频表征

Stage 1 使用冻结的 V-JEPA2.1 ViT-g/384 target encoder \(E_J\)。`384` 是该预训练模型的图像输入分辨率，
不代表 Con1 新模块的 hidden width。

对真实转移和同帧无变化参考分别编码：

\[
Z_t^R=\operatorname{sg}\!\left(E_J([o_t,o_{t+10}])\right),
\]

\[
Z_t^0=\operatorname{sg}\!\left(E_J([o_t,o_t])\right),
\]

其中：

\[
Z_t^R,Z_t^0\in\mathbb{R}^{24\times24\times1408}.
\]

\(\operatorname{sg}\) 表示 stop-gradient。V-JEPA2 始终冻结，Con1 不重新训练视频 foundation model。

### 5.2 只做空间降采样，不做固定通道投影

固定 JL/Rademacher 投影会在学习开始前把 V-JEPA2 的 1408 维语义通道压到一个随机子空间。它虽然节省存储，
却也可能不可逆地丢失少量但与接触或动作方向有关的通道；若最终结果失败，也无法区分是 Change 方法失败，还是
随机投影先破坏了 teacher。因此正式方法删除固定通道投影。

我们只保留一个没有参数的空间算子 \(S\)，使 Stage 1 teacher 与 JEPA-WAM 的 \(8\times8\) future-query
网格具有相同的粗空间尺度：

\[
S(Z)=\operatorname{L2Norm}\!\left[
\operatorname{AvgPool}_{3\times3}
\left(\operatorname{L2Norm}(Z)\right)
\right].
\]

其中 \(3\times3\) non-overlapping average pooling 只把 \(24\times24\) 降到 \(8\times8\)，不改变 1408 维
JEPA 通道。pair 与 no-change 两条路径使用完全相同的预处理、pooling 和逐 token normalization：

\[
Y_t^R=S(Z_t^R),\qquad Y_t^0=S(Z_t^0),
\]

\[
Y_t^R,Y_t^0\in\mathbb{R}^{8\times8\times1408}.
\]

工程上不分别保存 \(Y_t^R\) 和 \(Y_t^0\)，而是在同一次预计算中直接保存下节定义的 fp16 displacement。
这样既保留全部 JEPA 通道，又把磁盘占用从两套 dense cache 降为一套；该缓存不包含任何学习参数。

### 5.3 Latent change，而不是 latent pair

如果直接把 \(Y_t^R\) 当作 teacher，表示中会保留大量在当前帧和未来帧之间没有变化的场景内容。模型可能依靠
背景、物体身份或任务共现关系预测动作，而不是利用真正的时序变化。

因此，我们以同帧编码作为 no-change reference，定义：

\[
\boxed{
\Delta Y_t[i]=Y_t^R[i]-Y_t^0[i]
}
\]

这里对每个 token 分别归一化后再相减，但不对完整 \(\Delta Y_t\) 再做整体归一化。于是：

- 方向描述该位置的 JEPA 语义响应如何变化；
- \(\|\Delta Y_t[i]\|\) 仍保留相对于 no-change reference 的偏离强度。

这个量应严格称为 **no-change-referenced semantic displacement**。由于 V-JEPA2 token 已经过全局上下文化，
它并不是光流，也不能声称完全消除了静态信息；它表达的是“真实视频 pair 相对于静止 pair 多出的 latent
变化”。

### 5.4 Spatial Change Resampler

稠密 displacement \(\Delta Y_t\in\mathbb{R}^{8\times8\times1408}\) 仍然包含较多视觉信息。我们用
Spatial Perceiver-style Resampler \(G_\psi\) 将它压缩成：

\[
\boxed{
B_t^R=G_\psi(\Delta Y_t)
\in\mathbb{R}^{16\times128}
}
\]

16 个输出 token 固定排列为 \(4\times4\) latent grid。网络规格为：

```text
input grid         8×8×1408
input projection   1408 → 512
latent queries     4×4, width 512
resampler depth    3
attention heads    8
FFN                SwiGLU, hidden width 2048
output projection  512 → 128
output             4×4×128
```

每个 resampler block 执行：

\[
Q_l' = Q_l+operatorname{CrossAttn}_{2D}
\left(\operatorname{RMSNorm}(Q_l),\operatorname{RMSNorm}(X_t)\right),
\]

\[
Q_l''=Q_l'+\operatorname{SelfAttn}_{2D}
\left(\operatorname{RMSNorm}(Q_l')\right),
\]

\[
Q_{l+1}=Q_l''+\operatorname{SwiGLU}
\left(\operatorname{RMSNorm}(Q_l'')\right),
\]

其中 \(X_t=W_{\mathrm{in}}\Delta Y_t\)。

### 5.5 “空间锚点”的准确含义

这里的空间锚点不是检测框、物体中心、手工关键点，也不是只允许读取一个局部窗口。

它指的是：16 个 learned latent queries 分别拥有一个固定的 \(4\times4\) 归一化二维坐标。输入的
\(8\times8\) JEPA patches 和输出 queries 都映射到同一个 \([-1,1]\times[-1,1]\) 坐标系，并通过二维
RoPE 进入 Q/K。

因此，第 \((r,c)\) 个 query 在整个网络中始终保留第 \((r,c)\) 个粗空间位置的身份。它仍可通过
cross-attention 读取全部 64 个输入 patches；锚点只提供空间拓扑偏置，而不是硬裁剪感受野。

这具有两个作用：

1. Change tokens 不会退化为 16 个任意置换的无序 slots；
2. latent self-attention 仍能建模相距很远的区域，例如被操作物体与目标容器之间的关系。

因此，\(4\times4\) grid 同时保留了粗空间定位与全局关系建模能力。

### 5.6 为什么选择 16 个 token、每个 128 维

token 数量决定空间粒度，channel 数量决定每个位置能够表达的变化类型。

- 少于 16 个 token 会过早破坏物体、机械臂和目标区域之间的粗空间关系；
- 16 个 token 恰好形成 \(4\times4\) 网格，并与部署侧 future context 自然对齐；
- 64 维可以形成较强瓶颈，但对于多物体关系、姿态变化和接触状态可能过窄；
- 128 维给予每个空间 cell 足够的语义容量，同时保持明显压缩；
- 256 维会显著增大 Change flow，并提高保留静态捷径的风险。

最终 \(B_t^R\) 共有 \(16\times128=2048\) 个标量，而输入 JEPA grid 为
\(8\times8\times1408=90112\) 个标量，相当于约 44 倍的总容量压缩，并且只保留 \(4\times4\) 的粗空间
分辨率。压缩由可学习 tokenizer 完成，而不是由随机通道投影预先决定。

### 5.7 Direct inverse decoder：让 Change 对动作可辨识

仅压缩 \(\Delta Y_t\) 不能保证所得变量对控制有用。网络可能优先保存容易表达的视觉变化，而忽略能够区分
专家动作的因素。

因此，Stage 1 引入一个仅训练期存在的 direct inverse decoder \(D_{\mathrm{inv}}\)：

\[
\widehat A_t=D_{\mathrm{inv}}(B_t^R),
\qquad
\widehat A_t\in\mathbb{R}^{10\times7}.
\]

其结构为：

```text
action queries      10 learned temporal queries
hidden width        512
decoder depth       4
attention heads     8
each block          temporal self-attention
                    + cross-attention to 16 Change tokens
                    + SwiGLU FFN
output              10×7 physical actions
```

训练目标为：

\[
\mathcal L_{\mathrm{inv}}
=\frac{1}{10}\sum_{h=0}^{9}
\operatorname{Huber}
\left(\widehat a_{t+h},a^*_{t+h}\right).
\]

关键限制是：\(D_{\mathrm{inv}}\) 只能读取 \(B_t^R\)。它不能访问当前图像、语言、任务 ID、proprioception、
noisy action 或 flow time。

这一限制防止 inverse decoder 绕过 Change bottleneck。例如，如果把 noisy action interpolant 输入 inverse
flow，那么在 \(\tau\rightarrow0\) 时输入本身已经接近专家动作；低 loss 可能来自复制动作，而不是
\(B_t^R\) 真正包含控制信息。Direct inverse decoder 消除了这条捷径。

Stage 1 联合训练 \(G_\psi\) 与 \(D_{\mathrm{inv}}\)，训练结束后删除 \(D_{\mathrm{inv}}\) 并冻结
\(G_\psi\)。此后 teacher 定义为：

\[
\boxed{
B_t^R=\operatorname{sg}\left(G_\psi(\Delta Y_t)\right)
}
\]

它的准确语义是：

> **inverse-dynamics-aligned spatial change tokens：来自真实视频 latent displacement，并被约束为保留可辨识
> 当前专家 action chunk 的信息。**

它仍不是严格因果 action effect。专家数据只提供观测到的动作—结果相关关系，并没有提供相同状态下的反事实
动作干预。

### 5.8 Change endpoint 标准化

冻结 Stage 1 后，在训练数据上统计每个 channel 的均值和标准差，并在 16 个空间位置间共享统计量：

\[
\overline B_t^R
=\frac{B_t^R-\mu_B}{\sigma_B+10^{-6}},
\qquad
\mu_B,\sigma_B\in\mathbb{R}^{128}.
\]

Stage 2 使用标准化后的 \(\overline B_t^R\) 作为 Change flow endpoint。共享 channel statistics 不改变
16 个 token 的空间身份，同时使数据 endpoint 与单位高斯 source 具有可比较的数值尺度。下文为简洁起见仍将
它记作 \(B_t^R\)。

---

## 6. 部署侧的预测未来入口：JEPA-WAM future queries

### 6.1 为什么必须保留作者的 R

普通 image-language prefix 只受视觉语言和动作训练目标约束，并没有被明确要求预测未来。如果 Stage 2 只使用
普通 prefix，那么 Change stream 需要从头学习未来，方法也无法证明它利用了 JEPA-WAM 已有的预测能力。

因此，Con1 同时保留：

\[
C_t=\operatorname{Prefix}_{\pi_{0.5}}(o_t,\ell),
\]

\[
R_t^J=\operatorname{FutureQueries}_{\mathrm{JEPA-WAM}}(o_t,\ell)
\in\mathbb{R}^{64\times2048}.
\]

\(C_t\) 是普通当前条件；\(R_t^J\) 是作者已经通过 V-JEPA2 current–future target 监督过的 64 个
future-query outputs。

部署时 \(R_t^J\) 只读取当前图像和语言，因此它可称为 **predicted future context**。但它没有读取候选动作，
所以不能称为 action consequence、反事实 rollout 或保证能够实现的目标状态。

### 6.2 从 8×8 future grid 到 4×4 Change context

我们使用作者 checkpoint 中原有且受 JEPA objective 约束的 alignment head：

\[
U_t^J
=\operatorname{reshape}
\left(\operatorname{Align}_{\mathrm{frozen}}(R_t^J),8,8,1408\right).
\]

不执行原 JEPA loss 所需的 \(8\times8\rightarrow24\times24\) 插值，因为插值不会创造新的空间信息。
随后固定地将相邻 \(2\times2\) cells 合并：

\[
C_t^J
=W_J\left(
\operatorname{AvgPool}_{2\times2}
\left(\operatorname{L2Norm}(U_t^J)\right)
\right),
\]

\[
C_t^J\in\mathbb{R}^{4\times4\times1024},
\qquad W_J:\mathbb{R}^{1408}\rightarrow\mathbb{R}^{1024}.
\]

固定 pooling 保持局部邻接关系，并使 16 个 future-context cells 与 16 个 Change tokens 共享同一个
\(4\times4\) 空间顺序。这里不再加入另一组 learned resampler queries，避免形成第三套彼此不一致的 latent
坐标。

### 6.3 为什么不直接监督 C 与 B 相等

Con1 不使用：

\[
\operatorname{Huber}(C_t^J,B_t^R).
\]

原因不是不希望二者相关，而是它们的定义并不相同：

- \(C_t^J\) 来自 action-agnostic、较长时的预测未来；
- \(B_t^R\) 来自真实 H10 transition，并经过 inverse objective 选择控制相关信息；
- 两者的 channel width、训练历史和时间跨度不同；
- 固定的 \(4\times4\) 顺序只提供空间对应，不意味着每个 channel 应逐元素相等。

强制直接回归会在 Action–Change interaction 发生之前，要求 JEPA future 独自解释动作，这与方法动机矛盾。
因此，\(W_J\) 只通过最终的 action-flow 和 change-flow objective 学习：模型自行决定如何保留、抑制或重组
\(R_t^J\) 中对联合生成有用的信息。

同时，\(C_t^J\) 只进入 Change stream，Action stream 不直接读取 \(R_t^J\)。这保证核心路径为：

\[
R_t^J\rightarrow C_t^J\rightarrow\text{Change}
\leftrightarrow\text{Action}\rightarrow\widehat A_t.
\]

如果 Action 能绕过 Change 直接读取 \(R_t^J\)，那么即使动作改善，也无法说明动作可辨识的 Change 变量发挥
了作用。

---

## 7. Stage 2：Action–Change CoFlow

### 7.1 两个共同生成的连续模态

Stage 2 将动作和变化都视作 rectified-flow 的生成终点。分别采样独立高斯噪声：

\[
\epsilon_A\sim\mathcal N(0,I),
\qquad
\epsilon_B\sim\mathcal N(0,I),
\]

并共享同一个 flow time \(\tau\)：

\[
A_\tau=\tau\epsilon_A+(1-\tau)A_t^*,
\]

\[
B_\tau=\tau\epsilon_B+(1-\tau)B_t^R.
\]

这里 \(\tau=1\) 是纯噪声，\(\tau=0\) 是数据终点。对应的目标速度为：

\[
V_A^*=\epsilon_A-A_t^*,
\qquad
V_B^*=\epsilon_B-B_t^R.
\]

张量规格为：

```text
A_τ : 10×32       π0.5 checkpoint-compatible action tensor
B_τ : 16×128      4×4 spatial Change tensor
```

LIBERO 的真实物理动作是前 7 维；保留 32 维 action interface 是为了兼容 π0.5 checkpoint，而不是把 padding
维度解释成真实控制变量。

### 7.2 Change stream 的初始表示

Change noisy endpoint 投影到 MMDiT hidden width 后，与预测未来上下文和空间位置相加：

\[
H_B^0
=W_B B_\tau+C_t^J+E_B^{\mathrm{spatial}},
\]

其中：

- \(W_B:\mathbb{R}^{128}\rightarrow\mathbb{R}^{1024}\)；
- \(C_t^J\) 提供 JEPA-WAM 从当前观测预测的未来信息；
- \(E_B^{\mathrm{spatial}}\) 保持 16 个 Change tokens 的 \(4\times4\) 空间身份。

Action 与 Change 共享相同的 flow-time embedding，使二者在同一个生成时刻推理；但它们拥有独立的
AdaRMSNorm 参数，因为动作与视觉变化的数值分布和语义不同。

### 7.3 为什么 Action 前 12 层保持原 π0.5 路径

在每个 ODE step，Action tokens 先经过原 Action Expert 的前 12 个 blocks：

\[
A_{12}=F_A^{1:12}(A_\tau;C_t,\tau).
\]

这部分保持 action-only 结构，负责把 noisy action、当前图像语言 prefix 与 flow time 整合成稳定的 motor
representation。随后才进入 Action–Change joint reasoning。

这样做并不是认为前 12 层“低级”、后 6 层“高级”，而是设置一个明确的功能分工：

- 前 12 层保留作者预训练策略的动作生成基础；
- 后 6 层让预测变化参与动作的高层精化；
- Change 不在动作尚无任何结构时直接支配全部 motor computation。

这里的“后 6 层”是网络深度上的插入位置，不是只在某个 flow-time 区间打开。对所有 \(\tau\)，Action 与
Change 都在后 6 层共同去噪；不存在手工 late-flow threshold。

### 7.4 真正的双模态 MMDiT

Con1 不使用普通 Cross-Attention。普通 Cross-Attention 通常把一个模态固定为 memory，另一个模态单向读取：

\[
A\leftarrow\operatorname{CrossAttn}(A,B).
\]

这种结构中，Change 不会根据正在形成的动作更新自身；它仍只是外部 condition。

Con1 使用 MMDiT-style joint attention。Action 和 Change 保留各自独立的 normalization、QKV、output
projection 与 FFN 参数，但在同一个 attention matrix 中双向交换信息。

在第 \(l\) 个 joint block：

\[
\widetilde A_l=\operatorname{AdaRMSNorm}_A(A_l,e_\tau),
\qquad
\widetilde B_l=\operatorname{AdaRMSNorm}_B(B_l,e_\tau),
\]

\[
(Q_A,K_A,V_A)=W_A^{qkv}\widetilde A_l,
\]

\[
(Q_B,K_B,V_B)=W_B^{qkv}\widetilde B_l.
\]

ordinary current prefix \(C_t\) 在该层提供只读的 \((K_C,V_C)\)。构造一次联合注意力：

\[
Q=\operatorname{Concat}(Q_A,Q_B),
\]

\[
K=\operatorname{Concat}(K_C,K_A,K_B),
\qquad
V=\operatorname{Concat}(V_C,V_A,V_B),
\]

\[
O=\operatorname{Softmax}
\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
\]

输出按 query 类型拆回：

\[
(O_A,O_B)=\operatorname{Split}(O),
\]

然后分别经过模态专属参数更新：

\[
A_l'=A_l+W_A^oO_A,
\qquad
A_{l+1}=A_l'+\operatorname{FFN}_A(A_l'),
\]

\[
B_l'=B_l+W_B^oO_B,
\qquad
B_{l+1}=B_l'+\operatorname{FFN}_B(B_l').
\]

attention mask \(M\) 只表达三个结构事实：

1. Action 和 Change queries 都可以读取 ordinary current prefix；
2. Action 与 Change 在 flow suffix 内双向可见；
3. current prefix 是只读条件，不能读取 noisy Action/Change tokens。

此外，JEPA future-query tokens 不作为 Action 的直接 KV memory；它们只能先形成 \(C_t^J\) 并进入 Change
stream。

这一结构的关键语义是：

- Action 根据当前预测变化调整自身；
- Change 根据当前正在形成的动作重新解释 JEPA future；
- 二者在每一层、每一个 flow time 中共同收敛，而不是先预测一个固定变化再单向修正动作。

因此，CoFlow 建模的是：

\[
p(A_t,B_t\mid o_t,\ell,R_t^J),
\]

而不是两个彼此独立的条件边缘分布。

### 7.5 与 π0.5 Action Expert 的参数关系

Action 主干沿用 π0.5 的 `gemma_300m` Action Expert：

```text
hidden width   1024
depth          18
MLP width      4096
heads          8
head dim       256
```

参数组织如下：

- VLM/SigLIP backbone、64 个 future-query embeddings 和原 JEPA alignment head 保持冻结；
- Action input/output projections 与 Action blocks 1–12 沿用作者 checkpoint 并保持冻结；
- Action blocks 13–18 从作者 checkpoint 初始化，作为 MMDiT 的 Action side；
- Change side 拥有独立的 6 个 blocks，不与 Action side 共享参数；
- Change 的 normalization、Q 和 FFN 可以由对应 Action blocks 初始化，以获得稳定的数值尺度，但之后独立学习；
- Change 的 K/V 参数不能原样复制 Action 权重，而以原权重尺度的 \(10^{-3}\) 小初始化；Change output
  projection 同样小初始化，避免随机 Change stream 在训练前就强烈改写预训练 Action attention；
- \(W_J\)、Change input/output projections、Change spatial embedding 为新增参数。

Action 和 Change 使用相同 attention matrix，并不意味着二者共享 QKV 或 FFN。MMDiT 的核心恰恰是：
**联合注意力、模态专属参数**。这样既允许跨模态推理，也避免强迫视觉变化和 motor action 使用相同特征变换。

小初始化不是 learned gate，也不在训练中人为关闭 Change。由于 joint softmax 中新增 Change keys 仍会改变
归一化分母，step 0 不保证与作者策略逐位相等；实现后必须用同一 observation、action noise 和 ODE schedule
报告初始化 Con1 相对 JEPA-WAM 的 velocity 与 action endpoint 偏差。目标是把相对偏差控制在约 1% 内，确认
开始训练时仍处于预训练 motor policy 的局部邻域。若明显超出这一量级，应先减小 Change K/V 与输出初始化
尺度，而不是增加 gate。

### 7.6 速度输出与目标函数

最后分别预测两个速度场：

\[
\widehat V_A
=W_A^{\mathrm{out}}\operatorname{RMSNorm}(A_{18}),
\]

\[
\widehat V_B
=W_B^{\mathrm{out}}\operatorname{RMSNorm}(B_6).
\]

损失只有两个 flow-matching 项：

\[
\mathcal L_{\mathrm{action}}
=\operatorname{mean}
\left(M_{\mathrm{valid}}\odot
\|\widehat V_A-V_A^*\|_2^2\right),
\]

\[
\mathcal L_{\mathrm{change}}
=\operatorname{mean}
\left(\|\widehat V_B-V_B^*\|_2^2\right),
\]

\[
\boxed{
\mathcal L_{\mathrm{stage2}}
=\mathcal L_{\mathrm{action}}
+\lambda_B\mathcal L_{\mathrm{change}},
\qquad \lambda_B=0.3
}
\]

\(M_{\mathrm{valid}}\) 只保留 LIBERO 的 7 个物理动作维度，避免 25 个 padding dimensions 以容易预测的零值
稀释控制误差。

这里没有额外 \(C_t^J\rightarrow B_t^R\) matching loss，也没有人为规定某个 Action token 必须对应某个
Change token。\(B_t^R\) 通过 change-flow endpoint 提供 future-privileged supervision；Action 与 Change 的
对应关系由联合速度场在数据分布上学习。

---

## 8. “蒸馏”发生在哪里

Con1 可以称为一种 **future-privileged latent change distillation**，但需要准确说明蒸馏对象。

teacher 不是另一个直接输出动作的策略，而是 Stage 1 从真实未来构造的 change endpoint：

\[
(o_t,o_{t+10})
\rightarrow E_J
\rightarrow\Delta Y_t
\rightarrow G_\psi
\rightarrow B_t^R.
\]

student 在部署时看不到 \(o_{t+10}\)，只能从当前输入得到 \(R_t^J\) 和 ordinary prefix，并从噪声联合生成：

\[
(\epsilon_A,\epsilon_B,C_t,R_t^J)
\rightarrow(\widehat A_t,B_t^P).
\]

真实未来知识通过 \(B_t^R\) 定义 Change flow 的数据终点，从而进入 Stage-2 joint distribution。这里蒸馏的
不是逐元素 feature equality，而是：

> **由真实未来定义的动作相关 change distribution，被蒸馏进一个只依赖当前观测的 Action–Change 联合生成器。**

这也解释了为什么部署输出 \(B_t^P\) 不要求与某一个训练样本的 \(B_t^R\) 人工硬对齐。Flow 学习的是条件
分布及其与动作的耦合关系。

---

## 9. 部署时的数据流

部署阶段每次重规划执行：

1. 当前图像与语言经过冻结 π0.5 JEPA-WAM，得到 ordinary prefix \(C_t\) 和 64 个 future-query outputs
   \(R_t^J\)；
2. \(R_t^J\) 经过冻结 alignment head、固定 \(2\times2\) pooling 和 \(W_J\)，得到预测未来上下文
   \(C_t^J\)；
3. 采样 \(\epsilon_A\in\mathbb R^{10\times32}\) 与
   \(\epsilon_B\in\mathbb R^{16\times128}\)；
4. 在同一 ODE schedule 中从 \(\tau=1\) 积分到 \(\tau=0\)；
5. 每一步中，Action 先经过原 blocks 1–12，再与 Change 在 blocks 13–18 双向联合推理；
6. 得到动作 \(\widehat A_t\) 和预测变化 \(B_t^P\)，执行 \(\widehat A_t\) 的有效物理维度。

部署路径中明确不存在：

- 真实未来观测；
- V-JEPA2 target encoder；
- Stage-1 inverse decoder；
- 训练 teacher \(B_t^R\)；
- tracker、point-flow estimator 或外部 world model；
- success image 或额外环境 rollout。

因此 Con1 保持 current-observation-only deployment。

---

## 10. 为什么不采用更简单的替代结构

### 10.1 不直接使用 raw JEPA pair

raw pair latent 中静态场景占比很高，动作模型可能依赖场景共现而不是时间变化。no-change reference 先把研究
对象从“两个时刻的联合表征”收缩为“相对于静止 pair 的额外语义变化”。

### 10.2 不把 R 直接拼给 Action

这只能说明额外 condition 是否有用，不能解释 JEPA future 如何转化为动作相关变化。Con1 强制信息先进入
Change，再通过联合生成影响 Action。

### 10.3 不使用普通 Cross-Attention

普通 Cross-Attention 把 Change 当作固定 memory，只有 \(B\rightarrow A\) 的单向读取。MMDiT 允许
\(A\leftrightarrow B\) 在生成过程中反复相互更新，更符合“未来变化必须结合正在形成的动作才能被解释”的
假设。

### 10.4 不使用 tracker 或人工 point flow

tracker 会把方法的变化语义预先限定为二维可见点位移，也引入额外模型和伪标签。Con1 希望变化坐标系由
V-JEPA2 semantic displacement 与 inverse action objective 自身形成，能够同时容纳位置、姿态、接触和对象
状态等不一定能被点轨迹完整描述的变化。

### 10.5 不加入 learned gate 或手工 flow-time threshold

gate 容易让新路径在训练早期关闭，也会把“什么时候应该使用 JEPA”变成额外机制。Con1 让 Change 在所有
flow times 的后 6 个网络层中参与联合推理，信息是否有用由同一个速度场学习。

### 10.6 Stage-1 decoder 不读取当前状态

当前图像、语言和 proprioception 本身可能已经足以在专家数据上猜测平均动作。允许 decoder 使用它们会使低
inverse loss 无法证明 \(B_t^R\) 保存了变化信息。Stage 1 因而刻意把问题限定为：仅从真实 latent change
能否识别对应的 action chunk。

---

## 11. 各变量的严格语义

### \(\Delta Y_t\)：no-change-referenced semantic displacement

真实视频 pair 相对于同帧 pair 多出的 V-JEPA2 latent 响应。它不是像素光流，也不是严格局部运动。

### \(B_t^R\)：inverse-dynamics-aligned spatial change tokens

由真实 H10 transition 构造、带 \(4\times4\) 粗空间拓扑、被 inverse objective 约束为保留专家动作可辨识
信息的训练期 privileged endpoint。

### \(R_t^J\)：predicted future-query representation

JEPA-WAM 根据当前图像和语言预测的 future queries。它接受过真实未来监督，但不依赖当前候选动作。

### \(C_t^J\)：predicted future context for Change

\(R_t^J\) 经冻结 alignment、固定 pooling 和可学习投影得到的 \(4\times4\) context，只进入 Change stream。

### \(B_t^P\)：generated predicted change

部署时从噪声与 Action 一起积分得到的 change endpoint。它不是已经发生的真实变化，也不是独立 future-image
prediction。

### \(\widehat A_t\)：continuous action chunk

与 \(B_t^P\) 在同一个联合 flow 中生成的 10-step action。JEPA 信息通过 Change stream 与 MMDiT 进入该
输出，而不是直接由 JEPA decoder 产生。

---

## 12. Con1 的学术命题

Con1 所研究的不是“JEPA 能否直接预测机器人动作”，而是更窄、更可解释的问题：

> **当 VLM 已经通过 JEPA objective 学会预测未来表征时，能否以一个 no-change-referenced、
> inverse-dynamics-aligned 的空间 Change 变量为接口，通过 Action–Change MMDiT 将预测未来转化为连续控制
> 中可用的联合生成信息？**

它试图修正 JEPA 在 VLA 中常见的使用方式：JEPA 不再只是训练 backbone 的辅助 loss，也不是一个脱离策略的
外部 planner；它所预测的未来首先进入可生成的 Change modality，再与 Action Expert 共同推理。

方法可以概括为三项连续设计：

1. **No-change-referenced action-identifiable change abstraction**：从 V-JEPA2 pair latent 中提取动作可辨识
   的空间变化 endpoint；
2. **JEPA-WAM predictive-context routing**：保留已有 future-query prediction，并限制其只进入 Change
   stream；
3. **Action–Change CoFlow through MMDiT**：以模态专属参数和共享联合注意力共同生成动作与变化。

---

## 13. 可以声称与不能声称的内容

若该方法最终有效，最准确的表述是：

> We retain JEPA-WAM's current-conditioned future queries as predictive context for a generated Change stream,
> supervise that stream with inverse-dynamics-aligned V-JEPA2 semantic displacement, and couple it
> bidirectionally with continuous action generation through MMDiT. This exposes future-predictive VLM
> representations to the Action Expert under current-observation-only deployment.

可以声称：

- 使用真实未来构造的 privileged Change endpoint 被蒸馏到当前条件联合 flow；
- JEPA-WAM 已有 future-query prediction 通过 Change stream 参与连续动作生成；
- Action 与 Change 在 MMDiT 中双向联合去噪，而不是单向 condition；
- Change tokens 保留显式粗空间拓扑，并受到 inverse-dynamics objective 约束。

不能声称：

- \(\Delta Y_t\) 是光流或完整消除了静态信息；
- \(B_t^R\) 是严格因果的动作后果；
- 16 个 Change tokens 分别对应固定物体；
- \(R_t^J\) 是针对候选动作预测的反事实未来；
- \(B_t^P\) 是必然会实现的真实未来；
- JEPA 本身直接生成了 motor command；
- Con1 已经包含在线适应、TTA 或 TTT。

---

## 14. 与后续 Con2 的自然接口

Con1 本身不做在线更新，但它为 Con2 留下了一个不需要人工标签的闭环变量。

执行 \(\widehat A_t\) 并获得真实后续观测 \(o_{t+10}\) 后，可以使用与 Stage 1 完全相同的冻结编码器、固定
空间算子、Change Resampler 和标准化统计量构造 achieved change：

\[
Z_t^{\mathrm{ach}}
=E_J([o_t,o_{t+10}]),
\qquad
Z_t^0=E_J([o_t,o_t]),
\]

\[
\Delta Y_t^{\mathrm{ach}}
=S(Z_t^{\mathrm{ach}})-S(Z_t^0),
\]

\[
B_t^{\mathrm{ach}}
=\operatorname{Norm}_B
\left(G_\psi(\Delta Y_t^{\mathrm{ach}})\right).
\]

于是部署时预测的 \(B_t^P\) 与真实执行后得到的 \(B_t^{\mathrm{ach}}\) 位于同一个 learned change coordinate
system 中。Con2 可以据此研究预测变化与实际变化之间的自监督校准，而无需 reward、tracker 或人工成功标签。

这条接口使两项贡献保持连贯：

```text
Con1：从预测未来中联合生成 Action 与 Change
Con2：从实际后果中校准预测 Change 与动作—变化关系
```

Con2 的在线学习不属于本 Con1 方法定义。
