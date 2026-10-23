# 实验目录

先读 `evaluation.md`，再决定每版要测什么。跑完把结果追加到 `results.md`。当前已有 v1-v21 的环境、几何、q 级基线和接口诊断记录；正式跨操作者/跨机器人测评仍未完成。

每次从 `_模板/实验计划.md` 和 `_模板/failure_cases.md` 复制到新目录：

```text
experiment/v1_<核心方法>/
├── 实验计划.md
├── config.json          实际配置，格式可按所用工具调整
├── environment.txt      环境与依赖版本
├── logs/                训练/优化日志
├── metrics.json         实际测评结果，未运行前不要填数字
├── figures/             目标与实际叠图、分组图
├── failure_cases.md
└── checkpoints/         实际训练后才有的权重
```

编号连续，名字表达改动，例如 `v1_normalized_vectors`、`v2_shape_conditioned_mapping`。这些只是命名示例，并未创建成实际运行版本。
