# v10：共享 R_L20 几何/运动学模型

## 模型角色

这是由 L20 URDF/meshes 和 MANO-R 拟合参数构成的**单个共享目标虚拟手**。不同操作者共享同一组 `beta_R`、尺度、刚体对齐和参考几何；动作状态以后由第一阶段输出，再进入 L20 机构映射。R 是独立的拟人手，不是包裹 L20 的机械外壳。

## 已构建部分

- MANO-R 固定参考表面：`r_l20_surface.obj`，778 顶点、1538 三角面。
- L20 运动学骨架：22 个 link、21 个 revolute joint、根为 `hand_base_link`。
- 独立关节状态 `q`：16 个，顺序见 `r_l20_model.json`；5 个 mimic 关系保留在完整关节状态中。
- 21 个语义点：关节框架点、腕部根点和五个末节局部指尖点，均输出到 `hand_base_link` 坐标系。
- 共享模型数组：`r_l20_model.npz`；结构和零姿态语义点检查：`validation.json`。
- 21 个目标/拟合点及误差表：`r_l20_landmarks.csv`。
- R-only MuJoCo 场景：`../v10_R_L20_shared_model_mujoco/r_only/r_l20_only.xml`，只显示虚拟手 R。
- R/L20 对照 MuJoCo 场景：`../v10_R_L20_shared_model_mujoco/mujoco_r_l20_v10.xml`，仅用于坐标、点对应和外观差异检查，不表示 L20 被嵌入 R。

## 验收

`validation.json` 检查了 URDF 树、独立 q 数量、mimic 数量、所有 mesh 存在性、零姿态 21 点重算和中间限位姿态有限性。零姿态点重算不是动态重定向性能指标。

R-only MuJoCo 场景已加载通过，包含 1 个 body、0 个 joint 和 1 个视觉 geom；它只显示独立的 R。可用项目中的交互查看器打开：

```powershell
& '.conda\envs\mano-l20-sim\python.exe' `
  'scripts\view_mujoco_scene.py' `
  --scene 'experiment\v10_R_L20_shared_model_mujoco\r_only\r_l20_only.xml'
```

如需将 R 与 L20 放在同一坐标系中比较，再打开 `mujoco_r_l20_v10.xml`；该场景是诊断用叠加视图，不是 R 的结构定义。

## 当前边界

模型定义了目标几何和 q 级运动学接口，未定义真实执行器 `u`，未接摄像头，未运行动作重定向，也未进行真实尺度、装配零位、接触或柔顺标定。
