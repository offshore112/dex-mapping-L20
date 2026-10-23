# v5：MANO-R 仿真几何拟合

状态：**已运行仿真几何拟合；没有使用真机，没有进行动作重定向或执行器控制。**

## 输入

- MANO：`MANO_LEFT.pkl`，左手，10 个 β 形状参数。
- L20：URDF 零姿态，根坐标系 `hand_base_link`。
- 目标点：腕部 + 五指各 4 点，共 21 点。关节中心来自 URDF FK，指尖和腕部来自 v4 网格参考点。
- MANO tip 顶点：拇指 744、食指 320、中指 443、无名指 554、小指 671。

## 拟合配置

保留两个结果：

1. **尺度基线**：β=0、MANO 局部姿态为零，只拟合尺度和刚体对齐。
2. **当前 MANO-R**：拟合 β、正尺度、刚体对齐和一个固定的 `theta_ref_hand_pose`。`theta_ref_hand_pose` 只描述 L20 的 URDF 零姿态，后续动作变化仍应由第一阶段产生；L20 mimic 和执行器约束没有加入本次几何拟合。

## 结果

| 运行 | 点 RMSE | 最大单点误差 | 尺度 |
|---|---:|---:|---:|
| β=0 尺度基线 | 35.52 mm | 78.18 mm | 1.5876 |
| β+尺度+`theta_ref` | 10.84 mm | 23.06 mm | 1.5992 |

当前选用第二行作为仿真 MANO-R。误差按 21 个点计算，且是相对 L20 URDF/CAD 目标的静态点误差，不是真实硬件误差，也不是重定向性能。拇指、食指和小指的残差相对较大，说明标准 MANO 拓扑与 L20 机械关节代理之间仍有结构差异。

完整参数、β、`theta_ref_hand_pose`、刚体变换、逐点误差和网格数据见 [`fit_output_pose`](fit_output_pose/fit_summary.json)。

## 查看仿真效果

- [静态查看器](fit_output_pose/mano_r_l20_viewer_v2.html)：不使用 Canvas 或 JavaScript，直接显示 MANO-R 的正面、侧面和俯视图，适合当前 in-app browser。原 Canvas 版本保存在 `mano_r_l20_viewer_canvas_v2.html`。
- [静态叠图](fit_output_pose/mano_r_l20_overlay.png)：红色为 L20 目标点，绿色为 MANO-R 拟合点，蓝色为 MANO-R 表面。
- [MANO-R 网格](fit_output_pose/mano_r_fitted.obj)：可导入 Blender、MeshLab 或支持 OBJ 的仿真工具查看。

打开查看器时如果浏览器阻止本地脚本，可直接查看静态 PNG，或者用本地静态文件服务器打开 HTML；页面本身不访问网络。

## 边界

这只是第一版仿真几何拟合。它没有证明 MANO-R 与 L20 运动等价，也没有验证真实外壳、装配零位、碰撞、接触、动力学、SDK 或执行器映射。真机阶段可以在保留本版仿真参数的基础上，新增物理标定版本。
