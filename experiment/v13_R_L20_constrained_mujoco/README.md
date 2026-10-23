# v13 受约束 R 的 MuJoCo 预览

```powershell
& '.conda\envs\mano-l20-sim\python.exe' `
  'scripts\view_mujoco_scene.py' `
  --scene 'experiment\v13_R_L20_constrained_mujoco\r_only\r_l20_only.xml'
```

这是独立 R-only 场景，不包含 L20 机械网格。v13 固定平展姿态并约束 β，静态点 RMSE 为 35.16 mm、最大 79.22 mm；该结果需要在 R→L20 q 映射实验中验证。
