# v22：个人动作驱动固定形态 R

状态：**已运行单操作者离线姿态迁移和 MuJoCo 播放。** 使用用户已有的 774 帧 v21 个人 MANO 序列，目标形态取已有 v12 R 候选，不改变个人拟合结果和旧版显示。当前不是实时摄像头控制，也没有训练网络。

## 直接查看

双击本目录的启动文件：

- `open_R.cmd`：只看蓝色固定 R，播放完整 774 帧动作。
- `open_comparison.cmd`：金色为个人手、蓝色为 R，分开放置。
- `open_overlay.cmd`：两只手的腕部重合，检查大小和动作差异。

启动文件使用项目已有 `.conda/envs/mano-l20-sim/python.exe`，无需切换当前 Conda 环境。鼠标拖动旋转/平移/缩放；空格暂停/继续；关闭窗口退出。按原始时间戳循环播放，整段约 27.9 秒。

也可以从项目根目录运行：

```powershell
& .\.conda\envs\mano-l20-sim\python.exe scripts\view_fixed_r_mujoco_v22.py --view r-only
```

将 `r-only` 改为 `separate` 或 `overlay` 可切换视图。`--speed 0.5` 为半速播放；`--check-only` 只做数值核查；`--seconds 3` 可做有限时长 GUI 测试。

无需运行程序也能查看：

- [个人手与 R 并排动画](mujoco/separate/personal_to_fixed_r.gif)
- [R 单独动画](mujoco/r-only/personal_to_fixed_r.gif)
- [叠加动画](mujoco/overlay/personal_to_fixed_r.gif)
- [并排关键帧总览](mujoco/separate/comparison_keyframes.png)

GIF 覆盖整段时间，其中并排/R 单独预览以 12 FPS 采样，叠加以 4 FPS 采样；MuJoCo 播放器使用完整 774 帧并按时间戳选帧。总览使用与指定诊断帧最近的 GIF 预览帧，图内标注实际帧号。

## 输入、目标与映射

输入为 `experiment/v21_MediaPipe到MANO序列拟合/my_mano_sequence_v21_all.json`，物理左手、米、弧度，原始时间戳保留。目标为 `experiment/v12_R_L20_canonical_human/r_l20_model.json` 中的固定 `beta_R=0`、`scale_R=1.4920215577` 与固定旋转。该 R 是自然人手候选，与 L20 的联系保留在已有目标模型中；不是 L20 外壳或机械网格。

- 源手：使用个人 `beta_i` 和 `scale_i=1.0776804092` 重建。
- R：始终使用目标 `beta_R` 和 `scale_R`，不从输入 beta 替换目标形态。
- 动作基线：`theta_R(t)=theta_i(t)`，即相同左手 MANO 表示下的局部姿态直接迁移。
- 根部处理：两只手分别减去自己的 MANO 腕部位置，再用同一个固定旋转和目标腕部锚点放置。输入拟合的 global_orient/transl 只用于源重建一致性核查，不作为物理腕部运动传入 R。
- 分开视图增加的平移只用于显示，不参与映射和误差计算。

这一步只验证个人动作是否能够驱动一只固定形态的 R。没有额外时间滤波、捏合任务优化、L20 限位/mimic、执行器 u、接触或碰撞约束。

## 运行结果与边界

指标和定义见 [metrics.json](metrics.json)，配置/输入摘要见 [config.json](config.json)。

| 检查项 | 本次结果 |
|---|---|
| 输入/输出帧数 | 774 / 774，无丢帧 |
| 原序列时间跨度 | 27.9024 秒 |
| 目标 beta | 所有帧固定为 beta_R=0 |
| 目标腕部锚点偏差 | 0 m |
| 源重建与 v21 存储点最大绝对差 | 1.39e-17 m，说明 scale_i/旋转/平移约定核对通过 |
| 源/R 骨方向 RMSE | 1.1664°，全 20 条骨段、全帧 |
| 全 10 组指尖对距离平均绝对差 | 0.034857，分别除以各自掌部尺度后的无量纲差异 |
| 拇食指/拇中指对平均归一化距离差 | 0.023130 / 0.029900 |
| MuJoCo 蒙皮与生成顶点一致性 | 三种视图均核查全部 774 帧，最大绝对差约 7.27e-9 m |

以上是源手与目标 R 的动作保持诊断，不是真值重建精度或跨操作者泛化结果。相同姿态在不同骨长下不会自动产生相同的指尖关系；当前源序列本身也未提供表面接触真值。

最大局部姿态参数单步变化约 0.443704 rad，直接迁移会保留源序列中的变化；没有宣称抖动已消除。指标中姿态参数变化率是轴角参数分量差分，不能解释成真实关节角速度。

MuJoCo 使用 [visual skin 与 mocap body](https://mujoco.readthedocs.io/en/stable/XMLreference.html#deformable-skin) 播放实际 MANO 顶点：只更新数据中的 mocap 位置，不修改编译后的网格，不推进物理。R-only 场景为 1 个 skin、778 个 mocap body、0 个 joint、0 个碰撞 geom。它是可变形表面的运动学显示，不能用来做物理抓取验证。

## 重跑

```powershell
& .\.conda\envs\mano-l20\python.exe scripts\transfer_personal_hand_to_r_v22.py `
  --input experiment\v21_MediaPipe到MANO序列拟合\my_mano_sequence_v21_all.json `
  --mano E:\dex\MANO\mano_v1_2\mano_v1_2\models\MANO_LEFT.pkl `
  --model experiment\v12_R_L20_canonical_human\r_l20_model.json `
  --output experiment\v22_personal_fixed_r

& .\.conda\envs\mano-l20-sim\python.exe scripts\view_fixed_r_mujoco_v22.py `
  --view separate --render --fps 12
```

重复使用同一个 v22 输出路径会覆盖该次运行的输出；更换输入或配置时使用新的子目录，保留对照。

`r_sequence_v22.json` 中 `target_landmarks_m` 为含腕部锚定的权威目标。进入旧 v20 接口时必须使用 `--use-existing-landmarks`，避免旧接口重新前向时遗漏腕部锚定。本次已生成并核对 [r_sequence_v20_interface.json](r_sequence_v20_interface.json)，其 774 帧目标点与 v22 完全相同；没有继续运行 L20 q 优化。

## 下一步

先观察个人手/R 对照中的动作与指尖关系，保留本版为直接姿态迁移基线；再比较带指尖关系约束的映射与时间滤波。实时摄像头→MANO→R 还需要流式姿态估计和延迟验证，不能将本版离线整段拟合当作实时控制。跨操作者数据与两阶段网络仍按项目主线后续推进。
