# v6：MuJoCo 静态导入

状态：**已通过 MuJoCo 3.16.0 的无窗口模型加载和渲染检查。**

本版把 L20 左手 URDF 和 v5 的 MANO-R 拟合网格放入同一个 MuJoCo 场景：

- L20 保留 URDF 的 22 个 body、21 个关节和 5 条 equality/mimic 关系；
- MANO-R 作为独立的蓝色视觉网格加入；
- MANO-R geom 的 `contype=0`、`conaffinity=0`，不参与碰撞；
- 根坐标系为 `hand_base_link`，单位为米，关节角为弧度；
- 当前 OBJ 是 v5 参考姿态的静态网格，尚未实现每帧 MANO 顶点更新。

## 检查结果

`mujoco_v6_check.json` 记录了：23 个 body（包含 MANO-R 视觉 body）、21 个关节、23 个 geom，以及 MANO-R 网格存在且碰撞关闭。生成了零姿态和食指 PIP=0.5 rad 的无窗口渲染图。

- [零姿态](mujoco_v6_zero.png)
- [食指 PIP=0.5](mujoco_v6_index_pip_0p5.png)
- [MuJoCo 场景 XML](mujoco_l20_mano_r_v6.xml)
- [检查 JSON](mujoco_v6_check.json)

由于 MuJoCo 对含中文路径的资源读取存在兼容问题，本运行目录使用 ASCII 路径 `v6_mujoco_static_import`，并复制了 L20 的 `meshes` 目录；原始 URDF、STL、MANO 文件没有修改。

## 当前边界

这是静态导入和坐标验证，不是摄像头闭环。v7 已在此场景基础上完成 MANO-R 的逐帧离屏预览；L20 的独立关节状态 `q` 与后续执行器指令 `u` 仍未进行动作重定向。

## 交互查看

在项目根目录打开 PowerShell，运行：

```powershell
& '.conda\envs\mano-l20-sim\python.exe' `
  'scripts\view_mujoco_scene.py' `
  --scene 'experiment\v6_mujoco_static_import\mujoco_l20_mano_r_v6.xml'
```

会打开 MuJoCo 窗口。鼠标左键拖动旋转，右键拖动平移，滚轮缩放；关闭窗口后命令返回终端。若要查看 v7 的第 9 帧，使用：

```powershell
& '.conda\envs\mano-l20-sim\python.exe' `
  'scripts\view_mujoco_scene.py' `
  --scene 'experiment\v7_mano_r_motion_preview\rendered\scene_frame_008.xml'
```

v7 的临时 XML、OBJ 和 `meshes` 文件夹必须一起保留，因为 XML 使用相对路径引用这些资源。交互查看只显示选定的一帧；完整 9 帧动画请打开 v7 目录中的 GIF。

查看器默认是冻结检查模式，不推进物理，因此手指不会因重力或无控制器而抖动。只有明确加入 `--simulate` 才会推进 MuJoCo 动力学；当前场景没有动作控制器，仿真模式下关节可能运动，这不代表重定向结果。
