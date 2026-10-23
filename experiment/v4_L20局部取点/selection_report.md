# v4：L20 仿真局部参考点

状态：**已生成初始仿真点；未进行真机标定，未运行 MANO-R 拟合。**

本版的目的，是为仿真阶段固定一套可重复的 L20 腕部和指尖参考点。点来自本地 URDF 引用的 STL，不代表真实装配后的解剖腕点或实际接触中心。

## 坐标约定

- 根坐标系：`hand_base_link`。
- 单位：米；关节角单位：弧度。
- 每个 `local_xyz_m` 都位于该记录中的 `link` 坐标系。
- `zero_pose_root_xyz_m` 是 URDF 零姿态下通过 FK 变换到 `hand_base_link` 的位置。
- URDF 中这些网格的 visual origin 为单位变换，且没有显式 mesh scale。

## 选择规则

- `wrist_ref`：`hand_base_link.STL` 局部 `z` 最小端、厚度 0.1 mm 内三角面的面积加权中心。它表示仿真安装/掌部参考，不声明为解剖学腕点。
- 五个 `*_tip`：对应 `*_distal.STL` 局部 `z` 最大端、厚度 0.5 mm 内三角面的面积加权中心。它表示远端端盖中心，不声明为指腹接触中心。
- 不使用包围盒最大值作为点；点选择由脚本重新计算，避免手工复制时改变坐标。

## 初始点（局部坐标）

| 语义点 | link | x (mm) | y (mm) | z (mm) |
|---|---|---:|---:|---:|
| wrist_ref | hand_base_link | 0.082 | 0.005 | 0.000 |
| thumb_tip | thumb_distal | -13.838 | -1.055 | 31.238 |
| index_tip | index_distal | -15.222 | -0.006 | 22.826 |
| middle_tip | middle_distal | -15.222 | -0.006 | 22.826 |
| ring_tip | ring_distal | -15.222 | -0.006 | 22.826 |
| pinky_tip | pinky_distal | -15.222 | -0.006 | 22.826 |

未四舍五入的米制坐标、网格哈希、端盖面积和零姿态根坐标保存在 [`l20_r_reference_points_v4.json`](../../robots/L20_left/l20_r_reference_points_v4.json)。脚本 [`define_l20_reference_points.py`](../../scripts/define_l20_reference_points.py) 可以重新生成该文件。

## 使用边界

这组点可用于初始的 MANO-R 平移、旋转、尺度和 β 几何拟合。拟合时要在 MANO 一侧选择与这里语义一致的腕部和指尖点；如果研究目标转向接触力学或捏合接触中心，应另行增加 `contact_point`，不要覆盖本版 `tip`。

本版尚未验证网格与真实外壳的差异、装配零位、接触材料厚度、柔顺变形或 SDK 映射。接入真机时可以保留这组仿真点作为 CAD 基准，再增加一版物理标定点。
