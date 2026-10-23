# v14：R→L20 约束优化基线

本版本固定 v13 R，使用三个可复核的合成目标：平展张开、拇食指捏合、拇中指对指。优化变量是 L20 的 16 个独立关节状态 q，5 条 mimic 由 URDF 展开；没有定义执行器 u。

结果详见 `metrics.json`。20 mm 指尖距离只作为本轮 smoke diagnostic，不是冻结的正式任务阈值。目标来自固定 MANO 姿态向量，不是摄像头数据。

| action | landmark RMSE (mm) | max (mm) | target pair (mm) | actual pair (mm) | pair <=20 mm | mimic residual (rad) |
|---|---:|---:|---:|---:|---|---:|
| open | 22.897 | 37.393 | — | — | — | 0.000e+00 |
| pinch_thumb_index | 26.611 | 63.761 | 0.463 | 63.878 | False | 0.000e+00 |
| opposition_thumb_middle | 27.066 | 56.766 | 0.800 | 55.841 | False | 0.000e+00 |

## L20 机构自身的任务特征对照

下面的诊断只优化对应拇指/手指的 q，其他手指固定在 q=0；它回答 L20 机构本身能否把指尖靠近，不代表 R 的全关键点映射已经成功。

| pair | R target distance (mm) | L20 task-only minimum (mm) | active q count | mimic residual (rad) |
|---|---:|---:|---:|---:|
| thumb_tip + index_tip | 0.463 | 7.172e-12 | 7 | 0.000e+00 |
| thumb_tip + middle_tip | 0.800 | 2.689e-12 | 7 | 0.000e+00 |

可视化 q 解：[拇食指 task-only](renders/thumb_tip_index_tip.png)、[拇中指 task-only](renders/thumb_tip_middle_tip.png)。图中蓝色是静态 v13 R，灰色是 L20 q 解；两者没有通过接触/碰撞约束，图像只用于检查姿态差异。
