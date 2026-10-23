# v20：离线 MANO 序列接口与 v18/v19 稳定性比较

本目录把输入统一为离线 MANO 序列，再通过同一个 L20 目标条件化适配器比较 v18 逐帧求解和 v19 整段联合优化。原有回放使用项目已有的固定合成序列作为接口 fixture；真实摄像头采集已经能输出 MediaPipe 21 点，但还没有完成 21 点到 MANO 参数的拟合。

## 真实摄像头检查

已使用电脑自带摄像头 `camera 0` 实际采集 `640×480` 图像。原始采集记录在 `real_webcam_raw_capture.json`；带模型的采集记录在同目录的 `real_webcam_mediapipe_capture_*.json`。当前输出是 MediaPipe 21 点，尚未被当作 MANO 参数输入。

补齐模型后可使用：

```powershell
& D:\conda\envs\linker_hand\python.exe scripts\capture_camera_mediapipe_v20.py `
  --hand-landmarker-model models\hand_landmarker.task `
  --output experiment\v20_离线MANO接口稳定性比较\real_webcam_mediapipe_capture.json `
  --camera-index 0 --seconds 8 --display
```

## 结论

- v18 最大/平均 q 单步：`0.979151/0.243898 rad`。
- v19 最大/平均 q 单步：`0.830350/0.228392 rad`。
- v19 相对 v18 的最大 q 单步变化差：`-0.148801 rad`。
- v18/v19 原始任务对平均归一化误差：`{'thumb_tip+index_tip': 0.001433947747650051, 'thumb_tip+middle_tip': 0.0011199113258234315}` / `{'thumb_tip+index_tip': 0.0014886810596372215, 'thumb_tip+middle_tip': 0.001109485685660406}`。

输入格式说明见 `offline_mano_sequence_v20.json` 的 metadata 和 `scripts/prepare_offline_mano_sequence_v20.py`。
