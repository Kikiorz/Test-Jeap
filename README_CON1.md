# Con1 历史方案说明

本文件原先描述把 JEPA transition grid 通过 deformable attention 直接写入 Action Expert 的方案。

该方案已经停止：小规模实验表明，直接 condition、动作 residual 和候选重排均没有可靠利用 JEPA transition，
因此继续增加 spatial readout 结构会偏离已经观察到的主要问题：

```text
JEPA representation 更准确，不保证 JEPA 更新方向能改善动作。
```

当前 tracker-free 统一方案、数学定义、部署因果顺序、实现状态和负结果见
[README_JEPA_TTT.md](README_JEPA_TTT.md)。新方案只使用 JEPA-WAM 自身的 predicted transition 与执行后
V-JEPA realized transition，不使用 tracker、point flow、光流或人工运动标签。

旧方案仍可由 Git 历史恢复；它不再代表 `feat/point` 的当前方法。
