# v12 标准 MANO 中性 R 的 MuJoCo 预览

这个目录只显示独立的共享虚拟手 R，使用 `beta_R=0` 和平展 MANO 姿态，不包含 L20 机械结构。

```powershell
& '.conda\envs\mano-l20-sim\python.exe' `
  'scripts\view_mujoco_scene.py' `
  --scene 'experiment\v12_R_L20_canonical_human_mujoco\r_only\r_l20_only.xml'
```

预览图：[r_l20_only.png](r_only/r_l20_only.png)

v12 是自然外观候选，不是已经完成的 L20 几何拟合。静态 21 点误差为 RMSE 35.52 mm、最大 78.18 mm；后续应通过 R→L20 的约束优化验证它是否能支持动作映射。
