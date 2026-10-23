# 离线摄像头/MANO 序列接口

v20 把摄像头检测器和后续 R→L20 映射拆成两个文件接口。摄像头程序只需要输出 MANO 参数序列，后续脚本负责固定的 MANO→R 归一化、L20 目标条件化和 v18/v19 比较。

## 输入格式

输入 JSON 顶层可以使用 `frames` 或 `sequence`。每帧至少包含：

```json
{
  "frame_id": 0,
  "theta_hand_pose": [0.0, 0.0, 0.0],
  "beta": [0.0, 0.0, 0.0],
  "global_orient": [0.0, 0.0, 0.0],
  "transl": [0.0, 0.0, 0.0],
  "timestamp_s": 0.0
}
```

实际文件中 `theta_hand_pose` 必须是 45 个弧度值，`beta` 是 10 个形状值；`global_orient` 和 `transl` 是可选的 3 个值。当前接口固定为左手，长度单位使用米。

若输入已经包含 `target_landmarks_m`，可以在接口回放时使用 `--use-existing-landmarks`；真实摄像头流程通常让脚本根据 MANO 参数重新计算 21 个点。

## 运行

```powershell
cd E:\lingqiaoshou\dex-mapping
& .\.conda\envs\mano-l20\python.exe scripts\prepare_offline_mano_sequence_v20.py `
  --input <camera_mano_sequence.json> `
  --mano E:\dex\MANO\mano_v1_2\mano_v1_2\models\MANO_LEFT.pkl `
  --model experiment\v12_R_L20_canonical_human\r_l20_model.json `
  --output experiment\v20_离线MANO接口稳定性比较\offline_mano_sequence_v20.json
```

然后使用同一输入比较 v18 和 v19：

```powershell
& .\.conda\envs\mano-l20\python.exe scripts\compare_offline_mano_v18_v19_v20.py `
  --mano E:\dex\MANO\mano_v1_2\mano_v1_2\models\MANO_LEFT.pkl `
  --model experiment\v12_R_L20_canonical_human\r_l20_model.json `
  --input <camera_mano_sequence.json> `
  --output experiment\v20_离线MANO接口稳定性比较
```

当前 v20 回放使用已有固定合成序列作为接口 fixture，因此只验证文件链路和映射稳定性，不代表真实摄像头效果。固定 MANO→R 变换也是基线，跨操作者形态网络尚未接入。

## 电脑自带摄像头状态

当前机器的 `camera 0` 已验证可以打开并读取 `640×480` 图像；原始采集记录在 `experiment/v20_离线MANO接口稳定性比较/real_webcam_raw_capture.json`。本机 MediaPipe 使用 Tasks API，模型文件为 `models\\hand_landmarker.task`。`scripts/capture_camera_mediapipe_v20.py` 在原始未镜像帧上检测，默认只镜像预览；输出的 `handedness` 是物理左右手，`handedness_raw_mediapipe` 保留 MediaPipe 原始标签。如果摄像头驱动本身已经输出镜像图像，再额外加 `--input-mirrored`。该脚本输出的是 MediaPipe 21 点；v21 已提供离线 21 点→MANO 拟合接口。

默认 `--seconds 8` 到时会自动结束；需要持续预览时加入 `--until-q --display`，此时只有按 `q` 才停止并保存 JSON。

## v21：MediaPipe 21 点到 MANO

`scripts/fit_mediapipe_to_mano_v21.py` 读取 v20 的 `world_landmarks_m`，只接受物理左手，先建立每帧腕部原点和掌面坐标系，再用校准帧估计一个共享的 `beta_i/scale_i`。随后冻结形态参数，拟合全序列的 `theta_hand_pose`、`global_orient` 和 `transl`。

输出仍使用每帧 45 维 `theta_hand_pose`、10 维 `beta` 和 3 维 `global_orient/transl`，可以直接作为 `prepare_offline_mano_sequence_v20.py` 的输入。输出中的 `source_landmarks_m` 和 `fitted_landmarks_m` 用于诊断，`wrist_translation_world_m` 和 `palm_basis_world_from_local` 用于追溯坐标变换。

v21 仍然只完成 MediaPipe→MANO。它不直接输出 L20 关节，也不把 L20 `mimic` 或驱动约束混入第一阶段；随后才进入共享 R 和 R→L20 q 的接口。
