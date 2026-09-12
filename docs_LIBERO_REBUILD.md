# LIBERO 重建手册（新机器一条命令）

2026-09-12 那次重建踩到的所有坑都固化在 `scripts/libero_rebuild.sh` 里。换到任何新的 Blackwell 机器后：

```bash
mkdir -p /dev/shm/ts_jepa && cd /dev/shm/ts_jepa
# 把 scripts/{libero_rebuild,launch_frame_states,launch_anchored_features,launch_head_training,run_libero_arms}.sh 放进来
HF_TOKEN=hf_xxx bash libero_rebuild.sh env     # 环境（约 10 分钟）
HF_TOKEN=hf_xxx bash libero_rebuild.sh data    # 基座 + 数据集 + V-JEPA（约 10 分钟）
HF_TOKEN=hf_xxx bash libero_rebuild.sh cache   # 逐帧 states + anchored 缓存（约 1.5 小时）
HF_TOKEN=hf_xxx bash libero_rebuild.sh head    # Con1 head 12k 步（约 1.5 小时）
HF_TOKEN=hf_xxx bash libero_rebuild.sh arms    # 两臂各 4500 步（约 4 小时）
HF_TOKEN=hf_xxx bash libero_rebuild.sh eval    # LIBERO-Plus 环境
```

## 为什么每一步必须这样做

| 症状 | 根因 | 处理 |
|---|---|---|
| JAX 看不到 GPU | 仓库 pin `jax[cuda12]==0.5.3`，jaxlib 没有 sm_120 kernel | 固定 `jax/jaxlib/jax-cuda12-plugin/jax-cuda12-pjrt == 0.7.2` |
| `no kernel image is available`（V-JEPA conv3d） | torch 2.7.1+cu126 只编到 sm_90 | `torch==2.9.1+cu128`（含 sm_120） |
| `jax.experimental.layout 没有 DeviceLocalLayout` | orbax 0.11.13 与 jax 0.7.2 不配 | `orbax-checkpoint==0.11.20` |
| `'StepMetadata' object is not subscriptable` | orbax 0.12.x 改了 metadata API | 同上，别升到 0.12 |
| `Too many open files` → `Failed to launch ptxas` | 镜像 fd 软上限 1024 | `ulimit -n 65535` |
| 显存被占满、V-JEPA 拒绝加载 | 镜像自带 vLLM 服务占 4×89 GB | `supervisorctl stop vllm model-ui` + kill 残留 worker |
| `failed to map segment from shared object` | `/dev/shm` 是 `noexec` | venv 放 `/opt`，数据放 `/dev/shm` |
| `no version of torch==1.11.0+cu113` | 评测依赖 pin 了 cu113 索引 | 安装时加 `--extra-index-url https://download.pytorch.org/whl/cu113` |
| `'NoneType' has no attribute 'eglQueryString'` | 缺 `libEGL.so.1` | `apt-get install libegl1 libgles2 libglvnd0` |
| HF 限流（1000 req/5min） | 匿名拉 1699 个文件 | 用 `HF_TOKEN`，断点续传 |

## 关键路径与耗时（4× RTX PRO 6000 Blackwell）

| 阶段 | 实测 |
|---|---|
| 逐帧 V-JEPA states（1693 episodes / 273,465 帧，4 卡） | **55 分钟**（~85 帧/秒） |
| anchored 缓存（r + z，1693 episodes） | **28 分钟**（69 GB） |
| Con1 head 12k 步 | **87 分钟**（held-out `delta_nmse` 1.000 → 0.658） |
| 两臂各 4500 步 | 约 4 小时 |
| LIBERO-Plus L5 三类评测 | 约 2.2 小时 |

## 持久化提醒

`/dev/shm` 是内存盘，**实例重启即全丢**（2026-09-12 已发生一次）。要真正避免重复重建，需要一台**磁盘 ≥300 GB** 的机器把数据放盘上；否则每次重启都要重跑 `data → cache → head`（约 3 小时，脚本已就绪）。
