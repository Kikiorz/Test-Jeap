# Con1 + Con2：分支点算法说明（LIBERO / RoboTwin 共用）

这一版是**算法完整性冻结点**：以下所有部件都在同一份模型代码里，两个分支（LIBERO、RoboTwin）**只换数据配置和 latent 缓存**，模型与损失完全共用。

基座：官方 **JEPA-WAM PI0.5 40k**，基座参数通过 `BaseAndCon1HeadWeightLoader` 加载；`con1_*` / `con2_*` 不在基座里，保持各自初始化。

---

## 1. 三个新增部件

### Con1 核心：未来 latent delta + 门控 cross-attention

| 开关 | 值 | 作用 |
|---|---|---|
| `use_con1` | `True` | 总开关 |
| `con1_latent_dim` | 2816 | JEPA pooled latent 维度（VLM 分支用 2048） |
| `con1_train_action_layers_from` | 14 | 注入点：第 14 层之前做一次 residual retrieval |
| `con1_alpha_initial` | 0.3 | 门控初值；**α=0 时逐比特等于基座策略** |
| `con1_cross_attention_out_init` | 0.02 | 输出层小随机初始化（见第 4 节坑 1） |
| `con1_residual_budget` | 0.05 | 相对 hidden RMS 的硬预算；实测从不触发，仅作安全阀 |

`AnchoredDeltaHead(R_t, z_t) -> Δ̂_{t,1:10}`，`ActionDeltaCrossAttention(hidden, Δ̂)` 把它注入动作专家。

### Con2：delta 精修模块

| 开关 | 值 |
|---|---|
| `use_con2` | `True` |
| `con2_width` | 512 |

`Δ̃ = Δ̂ + F(Δ̂, z_t)`，`F` 的输出层零初始化 -> **step 0 严格退化为 Con1**。`F` 同时被两个目标监督：latent delta MSE（对真实未来 latent `Δ*`）与动作 flow loss。这正是"把 (Δz_pred, Δz_true) 送进函数 F，让它的梯度作用到动作上"的结构；上线时把 `Δ*` 换成**观测到的转移**即可复用为 TTT 载体。

### ④ 完整 VLM 上下文

| 开关 | 值 |
|---|---|
| `con1_vlm_context_tokens` | `True`（旧版 `con1_vlm_context` 是池化向量版，保持 `False`） |

`_con1_prefix` 把**整个 VLM 前缀**（全部图像 patch + 语言 token + state）连同 mask 交给 delta head；head 的检索注意力同时对 64 个 predictive token 与全部前缀 token 做 key/value，padding 用 mask 屏蔽。这样 delta 不再只看"64 个 query + 一个均值向量"，而是能看到构成图像 delta 的空间结构。

注意 `vlm_ctx_value` 用小随机（std 1e-2）而不是零初始化：零初始化会让 `vlm_ctx_key` 完全拿不到梯度（第 4 节坑 1 的同一个失败模式）。

---

## 2. 训练阶段与可训练参数

`mask_action_updates(freeze_before=14, depth=18, freeze_all=step < con1_stage1_steps)`：

| 阶段 | 步数 | 动作专家 | `action_out_proj` | `con1_*` / `con2_*` |
|---|---|---|---|---|
| stage 1 | `< 2000` | 全冻结（更新被清零） | 冻结 | 训练 |
| stage 2/3 | `>= 2000` | **只有 14–17 层**更新 | 训练 | 训练 |

冻结组（VLM 主干、SigLIP 视觉塔、`action_in_proj`、`time_mlp`、`vjepa_*`）在 checkpoint 里存成 **bfloat16**，可训练组存 **float32**——这是判断冻结是否生效的可靠依据（不要用原始 zarr 逐数组比，见第 4 节坑 2）。

---

## 3. 损失

```
L = flow_weight(step) * flow(delta_refined)        # 动作目标，主项
  + con1_delta_weight * delta_loss(delta, delta*)  # latent 目标
  + con1_residual_weight * ||correction||^2        # 修正幅度正则
```

`flow_weight` 从 2.0 cosine 衰减到 1.0（15000 步）；`con1_delta_weight=0.2`。动作专家与 Con1/Con2 模块共享同一个 flow 梯度，因此 **latent 分支是被动作目标直接监督的**。

---

## 4. 四个已验证的坑（避免重蹈）

1. **门控/输出层初始化太小 → 分支完全不动**。实测 `out_init=1e-3, α=0.05` 时，1000 步里 correction 能量恒为 `0.000000`、loss 不动。可用配置是 `out_init=0.02, α=0.3`；同样原因，VLM 上下文的 value 投影不能用零初始化。
2. **不要用原始 zarr 逐数组跨来源比较 checkpoint**。orbax 分片顺序不同会给出假的巨大差值（同一 checkpoint 与自身比是 0，换来源则处处不同）。判断冻结用 dtype，判断训练量用 orbax sharding 还原。
3. **Orbax 还原需要完整 4 卡 mesh**；1–2 卡会报 `sharding passed to deserialization should be specified`。训练、探针都固定 4 卡。
4. **stage 1 之前动作专家完全不训**；想练到后 4 层必须跑过 2000 步。

---

## 5. 分支需要各自提供的东西

| 项 | LIBERO 分支 | RoboTwin 分支 |
|---|---|---|
| 数据配置 | `LeRobotLiberoDataConfig`（`/workspace/artifacts/datasets/lerobot_libero`） | 该分支自己的 data config |
| latent 缓存 | `anchored_40k_features_v1`（1693 episodes，含逐帧 `R_t` 与 `z_t`） | 该分支自建缓存，字段名保持 `con1_current_latent` / `con1_future_latents` / `con1_future_valid` |
| delta head 初始化 | `con1_anchored_head_40k_20k_batch256_lr1e5/checkpoint_020000.msgpack` | 该分支自己的 head（先用 `train_head` 在本地缓存上训一个），或把 weight loader 换成 `CheckpointWeightLoader(base, missing_regex=".*con[12].*")` 让 head 随机初始化 |
| 完整配置名 | `pi05_libero_con1con2_ctx_40k` | 复制该配置，仅替换 data / assets / head 路径 |

模型、损失、阶段掩码、冻结规则**一行都不用改**。

---

## 6. 测试状态

`JAX_PLATFORMS=cpu pytest src/openpi/con1/ -q` -> **35 passed**，其中新增两项覆盖本轮改动：

* `test_vlm_context_tokens_attend_over_the_whole_prefix_and_respect_the_mask`：mask 掉的 token 改动后输出**逐比特不变**，未 mask 的 token 会改变输出，且 `vlm_ctx_key` / `vlm_ctx_value` 两个投影都拿到非零梯度。
* `test_vlm_context_tokens_trace_through_the_production_loss`：真实 `compute_con1_loss` 路径在开关打开时可前向并回传。
