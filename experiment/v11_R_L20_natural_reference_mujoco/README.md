# v11 自然参考 R 的 MuJoCo 预览

这个目录只显示 v11 的独立虚拟手 R，不包含 L20 机械结构。

```powershell
& '.conda\envs\mano-l20-sim\python.exe' `
  'scripts\view_mujoco_scene.py' `
  --scene 'experiment\v11_R_L20_natural_reference_mujoco\r_only\r_l20_only.xml'
```

预览图：[r_l20_only.png](r_only/r_l20_only.png)

v11 使用固定平展 MANO 姿态，视觉上更接近自然人手；但它的 L20 21 点静态误差仍为 RMSE 26.61 mm、最大 58.80 mm。这个候选用于确认 R 的独立拟人外观，不能作为已经完成的 L20 动态映射结果。
