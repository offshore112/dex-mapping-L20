# dex-mapping：跨人手尺寸与跨灵巧手机构的运动映射

本项目从你的[分享对话](https://chatgpt.com/share/6acf37ac-a0e0-83ec-84b0-c30e11a67063)整理而来，目录按照 [skJack/research-workflow](https://github.com/skJack/research-workflow) 创建。截至 2026-10-20，已完成 L20 文件运动学核查、R 候选与 q 级优化基线、真实摄像头采集、v21 单序列 MediaPipe→MANO 拟合和 v22 个人录制动作到固定 R 的姿态迁移/MuJoCo 播放。**实时摄像头到 R、跨操作者网络与正式测评尚未完成**。查看个人动作驱动 R 的入口见 [v22 说明](experiment/v22_personal_fixed_r/README.md)，最新状态见 [当前进度与下一步](docs/当前进度与下一步.md)。

## 先从这里开始

1. 阅读 [研究思路与新手指南](docs/研究思路与新手指南.md)。这是主文档，从基础概念讲到两阶段网络、实验设计和论文论证。
2. 按 [新手执行清单](docs/新手执行清单.md) 完成第一个里程碑，每次只推进一项。
3. 想了解原对话哪些说法需要修正，阅读 [来源与核查](docs/来源与核查.md) 和 [文献索引](related_work/文献索引.md)。
4. 已有 L20 左手 URDF 后，阅读 [URDF 到虚拟人手 R 建模指南](docs/L20左手_URDF到虚拟人手R建模指南.md)。它包含建模背景与历史计划；实际已运行结果以 [实验总表](experiment/results.md) 为准。
5. 当前电脑直接操作见 [独立 Conda 环境使用说明](docs/MANO与L20独立环境使用说明.md)，包含激活、验证和重跑命令。
6. L20 的正运动学、坐标和 mimic 检查见 [L20 检查说明](docs/L20正运动学坐标与联动检查.md)，离线交互页面和数值报告已生成。

`docs/研究思路与新手指南.html` 是主文档的离线阅读版，可用浏览器打开；执行清单、来源核查、数据说明、测评和文献索引也有对应 HTML，主文档可直接链接过去。`.md` 是可编辑的文字源文件。Markdown 是一种普通文本格式，`#` 表示标题，`-` 表示列表。修改文档后，运行 `python scripts/build_reading_view.py` 更新阅读版；脚本使用当前 Python 中已安装的 Markdown 库。

## 项目的主线

**不同尺寸的人手 → 尺寸映射网络 → 与目标机器人几何对应的虚拟人手 R → 受机构约束的映射 → 机器人执行器指令。**

“免逐对调参”和“未见机器人泛化”是研究目标，需要实验验证。R 是中间表示，不是保证与机器人运动完全等价的实体模型。

## 目录怎么用

```text
dex-mapping/
├── README.md                         给新手的入口
├── AGENTS.md                         项目目标、术语和协作规则
├── handoff.md                        进度与下一步
├── brainstorm.md                     讨论记录，最新的放前面
├── dataset.md                        数据来源、格式、处理与划分
├── docs/
│   ├── 研究思路与新手指南.md           主文档
│   ├── 研究思路与新手指南.html         离线阅读版
│   ├── 新手执行清单.md                分阶段操作与验收
│   └── 来源与核查.md                  对话依据、修正和证据状态
├── related_work/
│   ├── README.md                     文献存放方法
│   ├── 文献索引.md                   已核查的阅读入口
│   └── _模板/阅读记录.md              单篇阅读记录模板
├── 复现/
│   ├── README.md                     基线复现顺序
│   └── _模板/复现报告.md              单个基线报告模板
├── experiment/
│   ├── README.md                     实验存放方法
│   ├── results.md                    仅追加的结果总表
│   ├── evaluation.md                 统一测评规则草案
│   └── _模板/                        实验计划与失败样例模板
├── DataSet/
│   ├── raw/                          原始数据
│   ├── processed/                    处理后的数据
│   ├── splits/                       固定的训练/验证/测试名单
│   └── models/                       MANO 等模型文件
├── robots/
│   ├── README.md                     机器人模型与驱动描述要求
│   └── _模板/robot_descriptor.example.json
└── scripts/
    └── build_reading_view.py          生成主文档阅读版
```

上游的四个核心目录 `related_work/`、`复现/`、`experiment/`、`DataSet/` 与六个核心记录文件均保留；`docs/`、`robots/`、`scripts/` 是针对本课题补充的目录。

## 当前的第一件事

先运行 [v22 播放器](experiment/v22_personal_fixed_r/README.md)查看个人动作驱动固定 R 的效果，再检查指尖关系与抖动。v22 已把个人形态与目标形态分开；下一步比较任务约束与时间滤波，随后再接实时摄像头与跨操作者评估。旧 v20/v21 显示仍保留其沿用输入 beta 的历史行为。

以后开新对话，可以粘贴：

> 请先读 AGENTS.md 和 handoff.md，再读 docs/新手执行清单.md。用新手能理解的中文，推进当前第一个未完成的里程碑；先说明输入、输出、操作和验收方法，完成后更新 handoff.md。

创建日期：2026-10-14（Asia/Shanghai）。
