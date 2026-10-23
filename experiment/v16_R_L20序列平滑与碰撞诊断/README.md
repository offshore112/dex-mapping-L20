# v16：连续 R→L20 序列、时间平滑与碰撞诊断

本版本在 v15 功能特征 q 基线之上，生成一条 17 帧的固定合成序列：张开→拇食指捏合→拇中指对指。每帧使用同一组相对功能特征和 L20 URDF 限位/mimic 约束，并在相邻帧的 16 个独立 `q` 上加入归一化时间平滑项。

## 已运行结果

- 有平滑项 `smooth_weight=0.50`：17/17 帧求解成功，最大独立 q 单步变化 `0.434351 rad`，平均单步变化 `0.197362 rad`。
- 无平滑对照：最大单步变化 `0.866059 rad`，平均单步变化 `0.262775 rad`。
- 有平滑时平均骨方向 RMSE `16.812°`；拇食指和拇中指任务对的平均归一化距离误差分别为 `0.000300` 和 `0.000209`。
- 两种配置的限位违规均为 0，mimic 残差均为 0。平滑对照见 `smooth_ablation.json`。

## 文件

- `r_action_sequence_v16.json`：固定合成 MANO-R 目标序列。
- `feature_sequence_baseline/metrics.json`：带平滑的逐帧指标。
- `feature_sequence_baseline/sequence_q.json`：16 个独立 q 和展开后的完整 q 序列。
- `feature_sequence_no_smooth/metrics.json`：无平滑对照。
- `collision_diagnostic.json`：当前 URDF 碰撞网格的原始接触诊断。
- 动态 MuJoCo 观看文件位于 `experiment/v16_R_L20_sequence_viewer/renders/`：`sequence.gif` 是连续播放，`sequence_contact_sheet.png` 是 17 帧总览。相机使用正面视角，蓝色为动态 MANO-R，灰色为 L20 q 解。

## 当前边界

碰撞项没有加入优化。17 帧中最大原始接触数为 39，最大过滤父子相邻关系后的接触数仍为 26，说明当前 URDF 网格还不能直接充当干净的接触描述。未运行摄像头、物体交互、接触法向、真实驱动 `u` 或控制延迟；指尖距离仍不能称为抓取成功。
