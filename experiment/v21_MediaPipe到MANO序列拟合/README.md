# v21：MediaPipe 21 点到 MANO 序列拟合

状态：**已运行离线基线；不是跨操作者泛化结果。**

## 目的

v20 的摄像头文件只包含 MediaPipe 21 点，不能直接作为 MANO `beta/theta`，也不能直接作为 L20 关节目标。v21 增加一个两阶段离线拟合接口：

1. 在校准帧上估计一个操作者共享的 `beta_i` 和物理尺度 `scale_i`；
2. 固定 `beta_i/scale_i`，拟合每帧的 `theta_hand_pose`、`global_orient` 和 `transl`。

拟合在每帧腕部原点、掌面方向对齐的左手坐标系中进行。原始 MediaPipe 世界坐标、腕部位置、物理手性和时间戳均保留在输出中。

## 运行命令

```powershell
cd E:\lingqiaoshou\dex-mapping
& .\.conda\envs\mano-l20\python.exe scripts\fit_mediapipe_to_mano_v21.py `
  --input experiment\v20_离线MANO接口稳定性比较\real_webcam_mediapipe_capture_until_q.json `
  --mano E:\dex\MANO\mano_v1_2\mano_v1_2\models\MANO_LEFT.pkl `
  --output experiment\v21_MediaPipe到MANO序列拟合\mediapipe_mano_sequence_v21.json `
  --max-frames 48 --calibration-frames 16
```

随后可将 v21 MANO 序列接到现有固定共享 R 接口：

```powershell
& .\.conda\envs\mano-l20\python.exe scripts\prepare_offline_mano_sequence_v20.py `
  --input experiment\v21_MediaPipe到MANO序列拟合\mediapipe_mano_sequence_v21.json `
  --mano E:\dex\MANO\mano_v1_2\mano_v1_2\models\MANO_LEFT.pkl `
  --model experiment\v12_R_L20_canonical_human\r_l20_model.json `
  --output experiment\v21_MediaPipe到MANO序列拟合\r_normalized_v21.json
```

## 本次运行

输入为已有真实摄像头 MediaPipe 文件的一个固定子序列：48 帧，16 帧用于形态校准；22 个未检测到物理左手标签的前置帧被跳过。输出见 `mediapipe_mano_sequence_v21.json`，指标见 `mediapipe_mano_sequence_v21_metrics.json`。

- 形态校准中位掌长：101.84 mm；掌宽：67.35 mm。
- `beta_i` 范数：1.4624；`scale_i`：1.0791。
- MANO 关键点拟合 RMSE：7.377 mm；最大点误差：19.422 mm。
- 骨段长度平均绝对误差：3.195 mm；最大：12.113 mm。
- 这些数值是当前摄像头序列和损失配置下的拟合诊断，不能当作 MANO 真值精度、跨操作者结果或 L20 控制性能。

v21 输出随后通过现有 v18/v19 q 级接口回放：48/48 帧求解成功，限位违规 0，`mimic` 残差 0；v19 相对 v18 的最大 q 步长差为 -0.037784 rad。该回放只证明接口链路可运行，未加入碰撞、接触或执行器 `u`。

## 当前边界

- MediaPipe 世界点的深度方向和尺度仍依赖检测器输出，需用独立操作者/序列核查。
- 当前脚本只处理物理左手；右手需要单独镜像规则和测试，不能直接混入。
- 目前仍是逐操作者离线拟合，尚未训练跨操作者网络。
- v21 只生成 MANO 参数；共享 R 映射和 L20 `q/mimic/u` 仍是后续阶段。

## 生成显示结果

`prepare_offline_mano_sequence_v20.py` 本身是无窗口转换脚本。要查看 MANO-R 动作，可运行：

```powershell
& .\.conda\envs\mano-l20\python.exe scripts\render_v21_r_sequence.py `
  --mano E:\dex\MANO\mano_v1_2\mano_v1_2\models\MANO_LEFT.pkl `
  --model-manifest experiment\v12_R_L20_canonical_human\r_l20_model.json `
  --sequence experiment\v21_MediaPipe到MANO序列拟合\my_r_sequence_v21.json `
  --output experiment\v21_r_render_all_frames_ascii `
  --max-frames 0
```

输出为 `experiment/v21_r_render_all_frames_ascii/mano_r_contact_sheet.png`、`mano_r_sequence.gif` 和 48 个逐帧 PNG。`--max-frames 0` 表示不采样，按输入序列逐帧渲染。当前只显示 MANO-R 表面与 21 点；L20 q 级显示需要先完成 v18/v19 回放和 MuJoCo 渲染。

