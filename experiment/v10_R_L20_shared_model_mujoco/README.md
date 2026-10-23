# MuJoCo 静态场景：共享 R_L20

这是 v10 共享目标虚拟手的 ASCII 路径 MuJoCo 场景，用于避免 Windows 中文资源路径造成的加载问题。这里包含两个用途不同的场景：`r_only/r_l20_only.xml` 只显示独立的虚拟手 R；`mujoco_r_l20_v10.xml` 才是 R 与 L20 的对照叠加场景。

只查看独立的 R：

```powershell
& '.conda\envs\mano-l20-sim\python.exe' `
  'scripts\view_mujoco_scene.py' `
  --scene 'experiment\v10_R_L20_shared_model_mujoco\r_only\r_l20_only.xml'
```

```powershell
& '.conda\envs\mano-l20-sim\python.exe' `
  'scripts\view_mujoco_scene.py' `
  --scene 'experiment\v10_R_L20_shared_model_mujoco\mujoco_r_l20_v10.xml'
```

默认查看器是冻结检查模式，不推进物理。R-only 场景没有 L20 关节；R/L20 对照场景才包含 L20 关节机构。R 表面的碰撞已关闭，尚未进行动作重定向。

R-only 离屏检查见 [r_only/r_l20_only_check.json](r_only/r_l20_only_check.json)，R-only 图见 [r_only/r_l20_only.png](r_only/r_l20_only.png)。对照场景检查见 [mujoco_v10_check.json](mujoco_v10_check.json)，对照图见 [r_l20_v10_zero.png](r_l20_v10_zero.png)。
