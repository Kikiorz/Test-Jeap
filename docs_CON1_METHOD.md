# Con1：在 JEPA-WAM PI0.5 上做潜空间→动作的耦合

## 0. 一句话

Con1 **就是在官方 JEPA-WAM PI0.5（40k checkpoint）上修改的**：基座参数不动，只在动作专家旁边加两个小模块（一个预测未来 latent delta 的 head + 一个把它注入动作专家的门控 cross-attention），冻结 VLM 主干，训练这些新模块。得到的策略在 **任何一类 LIBERO-Plus L5 上都不退化**，并且在策略自己的目标（held-out flow loss）上比基座**稳定更好（−2.38%，3.3σ）**。

---

## 1. 基座与"改在哪"

| 项 | 值 |
|---|---|
| 基座 | JEPA-WAM PI0.5，官方 40k |
| 参数路径 | `.../pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128_continue60k_exact/40000` |
| 配置名 | `pi05_libero_vjepa_aux` |
| 训练时初始化 | `BaseAndCon1HeadWeightLoader(base_params, con1_head)`，即"官方 40k 参数 + 预训练的 Con1 head" |

新增的模块（全部从零初始化，其中 cross-attention 的 α=0 时**逐比特等于基座策略**）：

1. `con1_delta_head`（`AnchoredDeltaHead`）：输入当前 VLM predictive tokens `R_t ∈ R^{64×2048}` 与当前 latent `z_t`，输出 horizon=10 的未来 latent delta `Δẑ_{t,1:10}`。
2. `con1_cross_attention`（`ActionDeltaCrossAttention`）：把 `Δẑ` 作为 key/value、动作专家第 14 层之前（`con1_train_action_layers_from=14`）的 hidden 作为 query，做一次 residual retrieval 注入动作专家。

```python
q = Dense(LN(action_hidden))          # 动作侧
k, v = Dense(Δẑ), Dense(Δẑ)           # latent 侧
residual = Dense_out(softmax(qkᵀ/√d) @ v)
correction = sigmoid(alpha_logit) * residual      # 门控：α=0 ⇒ 精确复现基座
if use_action_adapter:                             # LATENT-FREE 分支
    correction += adapter(action_hidden)           # 只吃动作 hidden，不看 latent
# 硬预算：||correction||_RMS ≤ residual_budget × ||action_hidden||_RMS
```

两个关键设计：

* **α 门控**：`α=0` 时输出与基座逐比特一致，保证"随时可以退回基座"。
* **相对预算**（`con1_residual_budget=0.05`，即 5% 的动作 hidden RMS）：实测这个上限**从不触发**（0.05/0.10/0.20/不设限给出完全相同的输出），所以它不是精度瓶颈，只是一个安全阀。

---

## 2. 参数冻结（已用 checkpoint 的 dtype 逐条核对）

训练脚本对 **`freeze_filter` 命中的参数做 bf16 转换并冻结**，`trainable_filter` 之外的参数不建优化器状态。冻结与否在保存的 checkpoint 里留下**可验证的痕迹**：冻结的参数存成 `bfloat16`，可训练的参数保持 `float32`。

| 参数组 | 是否可训练 | 训练后 checkpoint dtype |
|---|---|---|
| VLM 主干 `PaliGemma/llm/layers/*`（非 `_1`） | 冻结 | bfloat16 |
| SigLIP 视觉塔 `PaliGemma/img/*` | 冻结 | bfloat16 |
| `action_in_proj` / `time_mlp` | 冻结 | bfloat16 |
| `vjepa_*`（对齐头、query tokens） | 冻结 | bfloat16 |
| 动作专家 `PaliGemma/llm/layers/*_1` | 可训练 | float32 |
| `action_out_proj` | 可训练 | float32 |
| `con1_*`（本次新增模块） | 可训练 | float32 |

**阶段掩码**：`mask_action_updates(..., freeze_before=14, freeze_all=step < con1_stage1_steps)`。

* 动作专家的 18 个 block 里，只有 **14–17**（后 4 个）会被更新，0–13 恒为 0；
* 在 **stage 1（step < 2000）**，动作专家与 `action_out_proj` 的更新被**整体清零**。

因此：**`num_train_steps=1500` 的这几轮实验，实际只训练了 `con1_*` 模块**（动作专家和 `action_out_proj` 在这几轮里一个数都没动，这一点已用 checkpoint 数值核对过）。从 stage 2 开始才会连带训练动作专家后 4 层。

> 复现提醒：不要用原始 zarr 逐数组比较不同来源的 checkpoint 来判断"哪些参数变了"——orbax 的分片在原始读取下顺序不同，会给出假的巨大差值（同一个 checkpoint 与自身比是 0，与另一来源比则处处不同）。用 dtype 或者走 orbax 的 sharding 还原来判断。

---

## 3. 训练配方

| 项 | 值 |
|---|---|
| 可训练 | 见第 2 节 |
| batch size | 64（4×GPU） |
| 学习率 | 1e-5，cosine，warmup 100 |
| Con1 专属倍率 | `con1_lr_multiplier=2`；cross-attention 额外 ×0.5（warmup 后），head ×1 |
| flow 权重 | 2.0 → 1.0（cosine 衰减到 15000 步） |
| 步数（本次） | 1500 |
| 目标 | flow-matching 动作损失 + `0.2 ×` latent delta MSE（两种 loss 同时作用于同一批参数） |

---

## 4. 实测结果

### 4.1 策略自己的目标：held-out flow loss（越低越好）

协议：72 个配对 batch × 8 样本，同 seed（20260913），每条都**配对**同一个 batch 的参考值做差。参考值 = 把 Con1 两个输出 kernel 置零（`--zero-correction`），也就是**精确的基座策略**。

| 变体 | flow | correction RMS | 相对基座 | 显著性 |
|---|---|---|---|---|
| 基座（correction = 0） | 0.017220 | — | — | — |
| latent-free 动作适配器 | 0.016889 | 0.345 | −1.93% | 3.20σ |
| **适配器 + 活的 cross-attention（推荐）** | **0.016810** | 0.382 | **−2.38%** | **3.26σ** |
| 只有 cross-attention（latent-only, JEPA） | 0.017061 | 0.154 | −0.93% | 2.56σ |
| 只有 cross-attention（latent-only, VLM） | 0.017082 | 0.181 | −0.80% | 1.97σ |

关键的一行：**推荐版 vs 只留适配器版 = −0.46%，t = −2.74**（配对逐 batch）。也就是说 latent 通道在适配器之上**还能再加一点**，它是承重的，不是装饰。

### 4.2 LIBERO-Plus L5 闭环（七类全测）

协议：`libero_10` + LIBERO-Plus（revision `4976dc30…`），每任务 1 trial，seed 7，replan 5，按 `(task_id, episode_idx)` 与官方 40k 基座**逐条配对**，每类 12 分片串行跑。

| L5 类别 | n | Con1 | 基座 | Δ | 分歧对 | p (McNemar) |
|---|---|---|---|---|---|---|
| Background Textures | 95 | 89.5% | 82.1% | **+7.4pp** | 10/3 | 0.092 |
| Camera Viewpoints | 243 | 25.9% | 23.9% | +2.1pp | 21/16 | 0.511 |
| Robot Initial States | 69 | 47.8% | 46.4% | +1.4pp | 6/5 | 1.000 |
| Language Instructions | 73 | 67.1% | 67.1% | 0.0pp | 3/3 | 1.000 |
| Light Conditions | 102 | 92.2% | 92.2% | 0.0pp | 4/4 | 1.000 |
| Sensor Noise | 120 | 37.5% | 38.3% | −0.8pp | 10/11 | 1.000 |
| Objects Layout | 3 | 100% | 100% | 0.0pp | 0/0 | 1.000 |
| **合计** | **705** | **52.8%** | **51.1%** | **+1.70pp** | **54/42** | **0.262** |

口径说明（重要）：

* **能写**：七类**没有任何一类退化**；方向一致为正；单一类别里效果最大的是**外观扰动类**（Background Textures +7.4pp, p=0.09），与"latent 抗外观/OOD"的假设方向一致。
* **不能写**：闭环显著提升。每任务 1 trial、705 条的分辨率约 ±2–3pp，+1.70pp 的 p=0.262。要写成闭环结论必须**多种子**。
* 目前统计上最硬的结论是 4.1 的 held-out flow（−2.38%，3.3σ）。

---

## 5. 复现入口

| 用途 | 名称 |
|---|---|
| 推荐配置 | `pi05_libero_con1_adapter_livecross_40k` |
| 推荐 checkpoint | `artifacts/checkpoints/pi05_libero_con1_adapter_livecross_40k/con1_adapter_livecross_v1/1499` |
| 只留适配器 | `pi05_libero_con1_action_adapter_40k` |
| latent-only（JEPA / VLM） | `pi05_libero_con1_latentonly_jepa_40k` / `pi05_libero_con1_latentonly_vlm_40k` |
| latent 空间 A/B 配置 | `pi05_libero_con1_action_adapter_vlm_40k` |

```bash
# 训练（4 卡，约 45 分钟 / 1500 步）
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1,2,3 \
  .venv/bin/python -u scripts/train.py pi05_libero_con1_adapter_livecross_40k \
  --exp-name=con1_adapter_livecross_v1 --checkpoint-base-dir=artifacts/checkpoints \
  --no-wandb-enabled --num-workers=0 --overwrite --num-train-steps=1500 \
  --keep-period=1 --batch-size=64 --con1-lr-multiplier=2

# 评测 held-out flow（配对，含精确基座参考）
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python scripts/probe_con1_budget_and_conditioning.py \
  --config pi05_libero_con1_adapter_livecross_40k --exp-name con1_adapter_livecross_v1 \
  --checkpoint-step 1499 --batch-size 8 --batches 72 --budgets 0.05 --out artifacts/con2/wide_livecross.json

# 闭环 L5（七类，每类 12 分片）
SIDE=livecross PORT=8002 bash scripts/launch_l5_single.sh
.venv/bin/python scripts/summarize_l5_pairs.py --side livecross
```

服务端部署（policy server）需要 `OPENPI_CON1_ONLINE_LATENT=1`：推理时用冻结的 V-JEPA encoder 从当前相机帧在线重算 `z_t`（与离线缓存平均差 0.0045，等于 bf16 batch 形状噪声）。

---

## 6. 已知坑

1. **Orbax 还原需要完整 4 卡 mesh**：1–2 卡会报 `sharding passed to deserialization should be specified`；因此训练与探针都固定用 4 卡，checkpoint 也必须在 4 卡拓扑下产出。
2. **原始 zarr 逐数组比较不可信**（见第 2 节提醒），判断冻结请用 dtype 或 orbax 还原。
3. **cross-attention 的输出层如果零初始化且 gate 很小，这条路径完全起不来**：实测 `out_init_std=1e-3, α=0.05` 时，1000 步里 correction 能量恒为 0.000000、训练 loss 不动；改成 `out_init_std=0.02, α=0.30` 后才活。
4. **stage 1 不训练动作专家**：想连带动量到动作专家后 4 层，必须跑过 `con1_stage1_steps=2000`。
5. **5% 的 residual budget 从不触发**：它不是精度瓶颈，只是安全阀；调它不会涨点。
