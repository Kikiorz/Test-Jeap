# SimpENV 交付说明（pi0.5 + Bridge：Con1 / Con1+Con2）

## 交付物

| 内容 | 位置 |
|---|---|
| 代码 | GitHub `Kikiorz/Test-Jeap` 分支 `feat/SimpENV` |
| 权重 | HuggingFace 私有仓库 `QRP123/simpenv-pi05-con1`（子目录 `armA` / `armB`） |
| 评测 harness | `examples/simpler_env/main.py`（4 任务 × 24 trials） |
| 一键评测 | `examples/simpler_env/run_eval.sh` |

## 评测任务数

SimplerEnv-WidowX 共 **4 个任务**，官方协议 **每任务 24 trials**（见 SimplerEnv 仓库 `scripts/octo_bridge.sh` 的 `--obj-episode-range 0 24`），即 **96 episodes/方法**：

1. `widowx_carrot_on_plate` — Put Carrot on Plate
2. `widowx_spoon_on_towel` — Put Spoon on Towel
3. `widowx_put_eggplant_in_basket` — Put Eggplant in Basket
4. `widowx_stack_cube` — Stack Green Block on Yellow Block

## 本地评测步骤

```bash
# 1) 代码 + openpi 环境
git clone -b feat/SimpENV https://github.com/Kikiorz/Test-Jeap.git ts_JEPA_simpenv
cd ts_JEPA_simpenv && uv sync

# 2) 权重
pip install -U huggingface_hub
hf download QRP123/simpenv-pi05-con1 --repo-type model --local-dir ./simpenv-ckpt

# 3) SimplerEnv 环境（官方仓库，需要 python 3.10/3.11 + 一块带 Vulkan 的 GPU）
git clone --recurse-submodules https://github.com/simpler-env/SimplerEnv.git
cd SimplerEnv && pip install -e ManiSkill2_real2sim && pip install -e .

# 4) 起 policy server（openpi 环境）
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/serve_policy.py \
  --env SIMPLER_ENV --port 8000 policy:checkpoint \
  --policy.config pi05_bridge_con1 --policy.dir /abs/path/simpenv-ckpt/armA

# 5) 跑 4 任务 × 24 trials（SimplerEnv 环境）
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json CUDA_VISIBLE_DEVICES=0 \
  python examples/simpler_env/main.py --task all --n-trajs 24 \
  --host 127.0.0.1 --port 8000 --log-dir ./simpler_eval
```

`VK_ICD_FILENAMES` 指向 NVIDIA 的 Vulkan ICD 是必须的，否则 SAPIEN 可能挑到软件 Vulkan（llvmpipe）而报扩展缺失。结果写入 `./simpler_eval/simpler_env_all_<时间戳>.json`。

## 动作/状态约定（已按 SimplerEnv 官方 widowx_bridge 对齐）

* 输入：第三人称 `obs["image"]["3rd_view_camera"]["rgb"]`（480×640，自动 resize 224）+ 8 维 Bridge state `[x,y,z,roll,pitch,yaw,0,gripper_openness]` + 语言指令。
* 输出：7 维 `[dx,dy,dz,dr,dp,dy,gripper]`；gripper 二值化（>0.5 视为张开 → +1，否则 −1），旋转 rpy 增量转轴角后送入环境。
* 每 `replan_steps=8` 步重规划一次（chunk 长度 10）。

## 训练配方（两臂唯一差别是 Con2）

| 项 | 值 |
|---|---|
| 基座 | 官方 `pi05_base` |
| 数据 | `IPEC-COMMUNITY/bridge_orig_lerobot`（LeRobot 格式；chunk-000 = 1000 episodes，单视角 image_0） |
| 配置 | `pi05_bridge_con1` / `pi05_bridge_con1con2` |
| 步数 / batch | 6000 / 32（FSDP 4 卡） |
| 学习率 | 5e-5 cosine，warmup 1000，EMA 0.999 |
| Con1 latent | pi0.5 自身 VLM 前缀池化（2048 维，离线缓存）；**不使用 JEPA 的 R_t** |
| Con2 | `Δ̃ = Δ̂ + F(Δ̂, z_t)`，zero-init，同时受 latent 与动作目标监督 |
