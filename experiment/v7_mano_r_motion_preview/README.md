# v7 MANO-R MuJoCo 动态预览

## 目的

在 v6 静态 MuJoCo 场景中检查拟合后的 MANO-R 网格是否可以逐帧更新和渲染。这个版本只验证可视化链路，不代表摄像头动作识别、MANO 参数估计或 L20 重定向已经完成。

## 输入

- MANO-R 拟合参数：`../v5_MANO-R拟合/fit_output_pose/mano_r_fit.npz`
- MANO-R 基准网格：`../v6_mujoco_static_import/mano_r_fitted.obj`
- MuJoCo 场景：`../v6_mujoco_static_import/mujoco_l20_mano_r_v6.xml`
- 预览序列：`mano_r_sequence.npz`

序列固定 v5 拟合得到的 `beta_R`、尺度、刚体对齐和参考姿态，只改变 MANO 原生 `hand_pose[3:6]` 的 x 轴角度，范围为 0 到 0.45 rad，共 9 帧。这是视觉烟雾测试，角度不是摄像头估计结果。

## 运行方法

```powershell
& '.conda\envs\mano-l20-sim\python.exe' -X faulthandler `
  'scripts\render_mano_r_mujoco_motion.py' `
  --scene 'experiment\v6_mujoco_static_import\mujoco_l20_mano_r_v6.xml' `
  --baseline-obj 'experiment\v6_mujoco_static_import\mano_r_fitted.obj' `
  --sequence 'experiment\v7_mano_r_motion_preview\mano_r_sequence.npz' `
  --output 'experiment\v7_mano_r_motion_preview\rendered'
```

脚本每帧写出一个 OBJ，并重新编译一个 MuJoCo XML；这样不修改 MuJoCo 模型内部的网格数组。临时 XML 会自动复制 v6 的 ASCII `meshes` 目录，以避免 Windows 中文路径和相对路径问题。

## 结果

`rendered/render_info.json` 显示 9 帧、每帧 778 个顶点均已渲染通过。输出包括：

- `rendered/frame_000.png` 到 `frame_008.png`
- `rendered/mano_r_motion.gif`
- 每帧对应的 OBJ 和 XML，便于复查

L20 关节状态固定为零姿态；MANO-R 网格为视觉几何，碰撞关闭。因此本结果不能用于证明 L20 的关节角、欠驱动联动或独立执行器命令已经得到映射。

## 崩溃记录与修复

第一次 v7 实现尝试直接改写 `model.mesh_vert`，并通过 `numpy.linalg.svd` 估计刚体变换；Windows Python 进程发生 `0xc06d007f` 崩溃。该失败保存在 [failure_cases.md](failure_cases.md)。已删除这条数值路径，改为逐帧 OBJ + MuJoCo 重新编译；单帧测试和 9 帧测试均正常退出。

## 下一步

把离线 MANO/关键点动作序列转换为统一的 `theta_R` 或关键点序列，再实现简单归一化、角度迁移和约束优化基线。之后才把 MANO-R 的姿态转成 L20 的独立关节/执行器空间，并分别记录 `q` 与 `u`。
