# v23：小拇指末节扭转诊断与修正候选

**最新结论：用户未接受当前动作自然性与弯曲网格，v23 未通过动作质量验收。** 小拇指局部修正和工程检查仍保留，但不能据此认为整体动作已恢复或能可靠驱动 L20。详见 [质量复核](run2_personal_mapping_test/quality_review/README.md)。

日期：2026-10-23（Asia/Shanghai）。**已运行原有 774 帧单操作者序列，保留 v21/v22。** 用户反馈为“小拇指只是在刚开始伸展时有问题，指尖的方向不对”。本版先检查静态模型、指尖语义和播放蒙皮，再处理开头明显的末节轴向扭转。

## 直接看结果

- **最新个人动作重复测试**：[run2 说明及播放入口](run2_personal_mapping_test/README.md)。已从原始 v21 输入重新修正/映射全部 774 帧，生成“个人拟合手 → R”并排动画并通过 30 秒 GUI 回放；金色为个人拟合手。以下 run1 入口仍保留为“旧 R / 新 R”对照。
- 双击 [open_R_v23.cmd](open_R_v23.cmd)：播放修正后的蓝色固定 R。
- 双击 [open_comparison_v23.cmd](open_comparison_v23.cmd)：金色是旧 v22 R，蓝色是 v23 R，二者使用同一个固定形态和尺度。
- [完整新旧对照动画](run1_preserve_points/comparison/old_new_fixed_r.gif)：覆盖约 27.9 秒，8 FPS 采样；播放器仍使用完整 774 帧。
- [开头 frame 0 三视图](run1_preserve_points/comparison/frame_0000_multiview.png)、[frame 30 三视图](run1_preserve_points/comparison/frame_0030_multiview.png)、[中段 frame 383 三视图](run1_preserve_points/comparison/frame_0383_multiview.png)。
- [未改变的中性 R](run1_preserve_points/comparison/neutral_fixed_r.png)。

播放器沿用 v22 显示程序：空格暂停/继续，鼠标旋转/缩放，关闭窗口退出。比较视图中的两只手都是目标 R，金色不再表示个人手。不要运行旧 v22 入口后把它误认为 v23。

## 定位结论

小拇指远端指间关节（distal interphalangeal joint，DIP）负责末节。弯曲改变手指弯向；沿指骨自身的旋转会改变指腹、指甲和末节表面的朝向。当前 21 点中的末节点与指尖，不能充分确定表面绕自身指骨的朝向。

本地核查结果如下：

1. MANO 左手的原生小拇指链是 `7→8→9`，父节点为 `0→7→8`；转换成项目 21 点后是 `17→18→19→20`。末节对应 45 维轴角参数的 Python 切片 `[24:27]`，没有把无名指和小拇指交换。
2. 指尖顶点 `671` 的蒙皮权重为 100% 归属原生关节 9；五根手指均核查对应一致。单帧与批量 21 点提取完全相同。
3. 中性前向与原始模板最大差为 `6.31e-9 m`；静态 R 没有额外的小拇指末节扭转。原始采集到 v21 的逐帧规范化点完全一致，掌部坐标基的行列式为 +1。
4. v22 播放表面与 MANO 输出一致。新旧 R 本次再次核查全 774 帧，蒙皮顶点最大差 `1.43e-8 m`。显示程序按输入的扭转姿态正常显示。
5. 开头 60 帧的源姿态末节轴向扭转绝对值平均 **112.54°**。frame 0 为 **+101.42°**，但当帧指尖中心距观测仅约 **1.15 mm**；所以此前全手位置 RMSE 和小拇指位置误差较小，不能排除此类表面方向异常。

扭转数值按个人形态的中性 DIP→指尖方向做 swing–twist 分解，零点是 MANO 中性姿态；它不是独立测得的解剖关节角。点拟合的多解性和缺失的末节扭转约束，解释了本次异常；真实指甲朝向仍没有图像真值支持。

## 修正方法与保留范围

`run1_preserve_points/` 是交付的修正候选。先把小拇指 DIP 旋转分成改变指骨方向的 swing 与沿指骨的 twist，从零扭转附近初始化，然后只重新求该 DIP 的三个旋转参数。拟合目标使用**旧 v21 已拟合的 21 点**，不新增标签、不改变源手尺寸。选择绝对扭转更小且保留原点的解；若不满足每点最大位移 `0.05 mm` 的数值保护条件则保留原帧。该容差是工程数值保护，不是视觉测量精度或正式任务阈值。本次 774 帧均找到满足条件的解。

固定不变的内容包括：个人 `beta_i/scale_i`、其余 42 维局部姿态、整体旋转/平移、原始检测点、帧 ID、时间戳，以及目标 `beta_R=0`、`scale_R=1.4920215577` 和腕部锚点。没有额外滤波、网络训练或 L20 q/u 求解。

然后复用 v22 的姿态迁移程序，把修正后的源姿态应用到同一固定 R。`fixed_r/` 内文件名和方法字段保留 v22，表示复用 v22 迁移算法；其输入明确指向本次 `mano_sequence_v23.json`，不是旧序列。

| 本次固定序列诊断 | 旧版 | 修正候选 |
|---|---:|---:|
| 开头 60 帧源 DIP 扭转绝对值均值 | 112.54° | 8.21° |
| frame 0 源 DIP 扭转 | +101.42° | +2.01° |
| 全序列源 DIP 扭转绝对值均值 | 93.99° | 15.96° |
| 相对 MediaPipe 的全 21 点 RMSE | 3.468683 mm | 3.468693 mm |
| 小拇指 DIP 相邻帧旋转变化最大值 | 25.566° | 25.551° |

修正前后，源手全部 21 点的最大位置变化为 **0.000995 mm**。固定 R 的小拇指指尖位置平均变化 **0.731 mm**、最大 **1.012 mm**，末段骨方向最大变化 **1.852°**；小拇指三个关节中心保持不变。源与目标 beta 不同，MANO 的姿态变形也不同，因此源点保持并不意味着目标 R 的点逐项完全相同。

frame 383 的正常中段姿态保留约 `−42.14°` 的原扭转解，位置几乎不变；没有按“开头若干帧”硬编码替换。完整序列的其他手指、固定旧诊断帧 `0/190/383/773` 均保留，新增 `5/30/60/120/250` 用于追踪本次现象。

统一的动作保持诊断也没有只挑变好的项：源/R 全骨方向 RMSE 为 `1.1664°→1.1534°`；全 10 对指尖归一化距离 MAE 为 `0.034857→0.035817`，略增。本版解决的是表面姿态歧义，不代表任务保持全面改善。

## 保留的失败对照

本目录根部的 `metrics.json`、`mano_sequence_v23.json`、`audit_geometry.npz` 和 `script_run0_zero_twist.py` 属于 **run0 强制零扭转**。该方案消除了扭转，但即使微调 swing，源指尖最大仍偏移 `5.184 mm`，frame 383 由 `0.435 mm` 的观测误差增至 `5.601 mm`。没有采用。不能把根目录 run0 文件当作当前修正序列；播放器已固定指向 run1。

## 复现与核查

在项目根目录运行。第一次命令如目录已有 `metrics.json` 会拒绝覆盖；要重跑请使用新输出子目录。

```powershell
& .\.conda\envs\mano-l20\python.exe scripts\audit_pinky_distal_v23.py `
  --mode preserve_points_min_twist `
  --output experiment\v23_pinky_distal_twist\run2_repeat

& .\.conda\envs\mano-l20\python.exe scripts\transfer_personal_hand_to_r_v22.py `
  --input experiment\v23_pinky_distal_twist\run2_repeat\mano_sequence_v23.json `
  --mano E:\dex\MANO\mano_v1_2\mano_v1_2\models\MANO_LEFT.pkl `
  --model experiment\v12_R_L20_canonical_human\r_l20_model.json `
  --output experiment\v23_pinky_distal_twist\run2_repeat\fixed_r

& .\.conda\envs\mano-l20\python.exe scripts\verify_pinky_audit_v23.py `
  --run experiment\v23_pinky_distal_twist\run2_repeat

& .\.conda\envs\mano-l20-sim\python.exe scripts\render_pinky_audit_v23.py `
  --run experiment\v23_pinky_distal_twist\run2_repeat
```

只检查现有交付文件：

```powershell
& .\.conda\envs\mano-l20\python.exe scripts\verify_pinky_audit_v23.py
& .\experiment\v23_pinky_distal_twist\open_R_v23.cmd --check-only
& .\experiment\v23_pinky_distal_twist\open_comparison_v23.cmd --check-only
```

主要证据：[run1 指标](run1_preserve_points/metrics.json)、[运行配置](run1_preserve_points/config.json)、[模型语义核查](run1_preserve_points/native_model_check.json)、[时序诊断](run1_preserve_points/temporal_check.json)、[R 与显示验证](run1_preserve_points/comparison/validation.json)。

## 边界与下一步

本次只处理个人已有序列的小拇指末节，修正依赖“优先选择接近中性的较小扭转解”这一先验。其他手指的轴向旋转、真实指甲方向、接触/碰撞、跨操作者、流式摄像头与两阶段网络尚未验证。后续计划是在视觉拟合中直接加入末节方向/扭转约束，并用指甲或指腹的图像证据验证；不能把单纯的 21 点低误差视为完整表面姿态正确。
