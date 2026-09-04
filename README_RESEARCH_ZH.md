# JEPA-WAM × π0.5：研究方案与当前实验结论

> 分支：`feat/point`  
> 基础模型：作者发布的 π0.5 JEPA-WAM，step `59999`  
> 更新日期：2026-09-04  
> 当前状态：**全部训练与验证已暂停；当前方法尚未证明能够改善机器人控制。**

## 1. 研究问题

JEPA-WAM 在训练期使用真实的当前—未来图像对监督 64 个 future queries，使 π0.5 能够仅根据当前观测和
语言预测一个 transition representation：

```text
U^P_t = f_JW(o_t, language).
```

但原模型默认不让 action tokens 读取这些 future queries：

```text
JEPA prediction ──► auxiliary JEPA loss
JEPA prediction ──X──► action generation
```

因此，JEPA loss 下降只表示预测表征更接近 teacher target，不代表动作更准确。我们的核心问题是：

> **如何让 JEPA 表达的任务转移与连续动作生成建立控制相关的对应关系，并在部署后利用真实交互继续校准
> 这一对应关系？**

这不是单纯把 JEPA latent 作为 action condition，也不是再训练一个外挂 world model。

## 2. 变量与部署边界

动作块长度固定为 `H=10`。训练样本应严格对应：

```text
(o_t, language, q_t, A*_[t:t+H-1], o_(t+H)).
```

执行前的预测转移为：

```text
U^P_t = Normalize(Pool_8(G_align(R_t))) ∈ R^(64×1408).
```

训练期或动作执行后的真实转移为：

```text
U^R_t = stopgrad(Normalize(Pool_8(E_J([o_t, o_(t+H)])))) ∈ R^(64×1408).
```

其中 `E_J` 是冻结的 V-JEPA。`U^R` 是学习式 current–future joint representation，不是人工光流、tracker、
point flow、成功图像或真实因果状态。

部署当前动作时只能使用 `o_t`。`o_(t+H)` 只有在动作已经执行后，才能用于下一次测试时更新。

## 3. 最终设想：CoFlow-JEPA

### 3.1 Con1：Action–Transition Co-Flow

Con1 的设想不是固定映射：

```text
U^P_t → Action.
```

而是让动作假设与转移假设在同一个 rectified-flow 过程中共同演化：

```text
(Gaussian action noise, predicted transition prior)
                       ↓ joint vector field
(expert action, action-consistent realized transition).
```

沿用 π0.5 的时间约定，`tau=1` 为 source，`tau=0` 为 target：

```text
A_tau = tau * eps_A + (1-tau) * A*
U_tau = tau * U^P   + (1-tau) * U^R

V*_A = eps_A - A*
V*_U = U^P - U^R
```

联合目标为：

```text
[V_hat_A, V_hat_U] = F_theta(A_tau, U_tau, prefix(o_t,l,q_t), tau)
L_CoFlow = L_action + lambda_U * L_transition.
```

网络候选采用双流 joint-attention：action 与 transition 使用各自的 Norm、QKV 和 FFN，但允许双向交换
信息。Action→Transition 的影响随 `1-tau` 平滑增强，避免纯噪声动作污染 transition stream。

### 3.2 Con2：Sparse Inverse TTT

执行动作块后，机器人天然得到无奖励监督：

```text
(U^R_t, A_exec_t).
```

设想中的 Con2 使用 sparse inverse objective，从真实 transition 中识别对已执行动作有辨识力的 token，并
只更新 Con1 中同一组 Transition→Action coupling：

```text
generate → execute → observe U^R
         → inverse-dynamics update
         → generate the next action chunk.
```

这不是 RL：不使用 reward、value、advantage 或 policy gradient。它属于部署时 inverse-dynamics adaptation。
为避免模型只是更好地重建错误动作，最终还需要离线 meta-objective 验证“一次 inverse update”能降低同一
部署条件下另一个 query state 的 action loss。

## 4. 当前真正实现了什么

### 4.1 已实现

- 同状态多动作干预数据采集：精确恢复同一 MuJoCo 状态后执行 `base/+x/-x/+y/-y`；
- 冻结 V-JEPA 的真实 transition 提取与 JEPA-WAM current-only prediction 审计；
- 小型 action-conditioned consequence、inverse decoder 和 CoFlow core 的可证伪实验；
- 将 JEPA target 从作者约 `t+31` 调整到与动作块一致的 `H=10`；
- 从作者 checkpoint warm-start，只训练 64 个 future queries 与 alignment head；
- 可选的 training-only inverse-action loss，用于约束预测 transition 保留动作可解码信息。

最近一次训练目标是：

```text
L = L_JEPA-H10 + lambda_inv * L_inverse-action.
```

VLM、V-JEPA 和 Action Expert 均冻结。因此这属于 **action-aware future-query adaptation**，不是完整的
CoFlow 架构。

### 4.2 尚未实现

- 没有把 joint CoFlow blocks 接入 π0.5 Action Expert；
- 没有训练完整的 action–transition joint vector field；
- 没有部署时在线更新；
- 没有证明 TTT/TTA 能提高下一步动作或任务成功率；
- 没有运行以当前方法为对象的大规模 LIBERO-Plus rollout。

## 5. 核心实验结果

### 5.1 历史直接接口均没有成功

| 方法 | 关键结果 | 结论 |
|---|---:|---|
| 小型 Joint CoFlow | integrated active-action MSE `0.4381 → 0.4721` | 联合流比 matched fixed condition 更差 |
| additive transition guidance | `0.01452 → 0.01470` | 无改善 |
| inverse candidate rerank | `0.004733 → 0.005481` | 变差 |
| transition-energy gradient | 改善极小且 bootstrap 95% CI 跨 0 | 不可靠 |
| shared-adapter / inverse TTT gate | correct update 与 shuffled controls 无稳定差异 | 缺少 target specificity |

这些结果说明：

```text
R contains transition/action information
                 ≠
R is a usable local control objective.
```

### 5.2 同状态干预证明真实 transition 含有 action imprint

对 4 个 LIBERO-Spatial 任务各取 4 个状态，每个状态执行五个动作分支，共 80 条 transition：

```text
16 states × {base,+x,-x,+y,-y}.
```

同起点恢复前图像最大 MAE 仅 `0.00729/255`。动作间差异与结果信号为：

| 指标 | 数值 |
|---|---:|
| action pair RMS | `0.0994` |
| future-image pair MAE | `2.053` |
| realized V-JEPA pair distance | `0.02088` |
| held-out 五动作识别 top-1 | `70.0%`（随机 `20.0%`） |

因此，冻结 V-JEPA 的真实 current–future representation 确实含有较大局部干预的 action imprint。失败不能
简单归因于“JEPA 完全没有动作信息”。

### 5.3 H=10 future-query 严格 A/B 对照

使用同一批 320 个专家片段、同一 episode-disjoint split、相同初始化、200 steps：

| 训练 | held-out JEPA 距离（前→后） | inverse-action MSE（前→后） |
|---|---:|---:|
| 纯 H10 重训，`lambda_inv=0` | `0.132759 → 0.132256` | `0.156389 → 0.156073` |
| action-aware，`lambda_inv=0.1` | `0.132767 → 0.132558` | `0.156298 → 0.155716` |

inverse-action 项的方向符合预期：相对纯重训，最终 inverse MSE 额外降低约 `0.000357`，但幅度只有约
`0.23%`，同时 JEPA prediction 略差。单凭这一结果不能认为算法有效。

### 5.4 controlled-axis 迁移失败

将两种 H10 predictor 用于严格同状态干预初始状态：

| predictor | 到真实 matched transition | no-change persistence |
|---|---:|---:|
| 纯 H10 重训 | `0.167849` | **`0.052055`** |
| action-aware H10 | `0.168274` | **`0.052055`** |

两种 predictor 都显著差于 persistence；action-aware 版本还略差于纯重训。

使用完全相同且冻结的 consequence model 做五候选选择：

| desired prior | learned selection | 最佳固定方向 |
|---|---:|---:|
| 纯 H10 重训 | `0.166382` | **`+y: 0.165652`** |
| action-aware H10 | `0.166805` | **`+y: 0.166093`** |

两组都在 8 个验证状态中 7 次选择 `+x`，只有 1 次选择 `+y`。这仍是接近固定方向的偏置，不是可靠的
状态相关规划。

## 6. 当前判断

这轮实验区分了“代码 bug”“重新训练收益”和“算法收益”：

1. 梯度、参数过滤、checkpoint 保存与重新加载均已通过 smoke test，没有发现导致结论反转的代码错误；
2. 纯 H10 重训只有很小的 held-out 改善；
3. inverse-action 约束产生了极小的动作可解码性增益，但没有迁移到同状态控制干预；
4. current-only predictor 不接收候选动作，单专家动作监督只能强化行为相关性，无法识别反事实
   action–transition coupling；
5. 当前证据不支持继续延长相同训练，也不支持直接进入大规模 rollout 或 Con2。

因此最诚实的结论是：

> **V-JEPA 的 realized transition 含有动作印记，但当前 JEPA-WAM 的 current-only predicted transition
> 不是可靠的 10-step 局部控制目标。简单 horizon 对齐和 inverse-action regularization 均不足以解决这一
> 根本问题。完整 CoFlow-JEPA 仍是研究设想，而不是已经验证的方法。**

## 7. 暂停后的研究决策点

恢复研究前应先决定是否继续使用当前 V-JEPA joint target。合理选择只有两条：

1. **停止 CoFlow-JEPA 主线**：接受 current-only prior 不适合作为局部控制变量，重新定义更动作相关的
   transition representation；
2. **直接验证真正的 joint generation**：不再要求固定 `U^P` 单独承担 desired motion，只在严格同状态多动作
   数据上检验 joint flow 是否学会 noise-dependent、多模态的 action–transition coupling。只有它显著超过
   matched fixed-condition flow，才值得接入完整 π0.5。

不建议继续：增加更多 readout、gate、reranker、tracker、point-flow 标签，或只延长当前 future-query
训练。这些操作没有触及已观察到的主要失败原因。

## 8. 关键代码与远程产物

核心代码：

- `examples/libero/collect_action_interventions.py`：同状态多动作干预采集；
- `scripts/audit_openpi_jepawam_pi05.py`：JEPA-WAM predictor 与 V-JEPA target 审计；
- `scripts/fit_horizon_future_queries.py`：H10 future-query A/B 训练；
- `scripts/run_intervention_consequence_gate.py`：action sensitivity 与候选选择；
- `scripts/run_coflow_core_gate.py`：小型 CoFlow 与 fixed-condition 对照；
- `src/openpi/models/coflow.py`：当前最小 joint-flow 原型。

最新远程结果：

```text
/workspace/artifacts/outputs/h10_future_query_plain200_s271/
/workspace/artifacts/outputs/h10_future_query_action200_s271/
/workspace/artifacts/outputs/action_interventions_axis_spatial4x4_k5_d02_s233_predict_h10_query_plain200_s271/
/workspace/artifacts/outputs/action_interventions_axis_spatial4x4_k5_d02_s233_predict_h10_query_action200_s271/
/workspace/artifacts/outputs/intervention_axis_select_h10_query_plain200_s271/
/workspace/artifacts/outputs/intervention_axis_select_h10_query_action200_s271/
```

这些结果目录均保留；当前没有训练或评估进程运行，4 张 GPU 均为空闲状态。
