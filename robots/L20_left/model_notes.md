# L20 左手模型来源核查

日期：2026-10-14（Asia/Shanghai）。状态：**在线来源、本地 URDF 结构和网格存在性已核查；FK/仿真/真机未运行。**

用户本地目录为 `E:\lingqiaoshou\linkerhand-urdf-main\linkerhand-urdf-main\l20\left`，URDF 与引用的 22 个不同网格文件均存在，哈希见 [本地资源审计](local_resource_audit.json)。当前目录只放核查记录，不含原始 URDF、STL 或已验证的驱动描述。原始资源保持只读。

- 官方目录：[L20/left](https://github.com/linker-bot/linkerhand-urdf/tree/main/L20/left)。
- 文件：[linkerhand_l20_left.urdf](https://github.com/linker-bot/linkerhand-urdf/blob/main/L20/left/linkerhand_l20_left.urdf)。
- 本次读到 main 的完整 1204 行；commit SHA 待固定，本地文件哈希待计算。
- 仓库页面标注 Apache-2.0；正式保存模型时仍须留存适用许可文件与原始版权说明。
- 核查统计与每关节属性见 [urdf_source_audit.json](urdf_source_audit.json)。该 JSON 是来源审计，不是可直接控制机器人的 descriptor。

统计：22 个 link；21 个 revolute joint；5 条 mimic；施加这五条关系后有 16 个独立运动学坐标。四指 DIP=0.8917×PIP，拇指 IP=1.1619×MCP，offset 全为 0。未发现 transmission、ros2_control 与独立命名的指尖 link。

网格引用为相对的 `meshes/*.STL`。根参考系是 `hand_base_link`；仿真用腕部代理和五个末节局部指尖初始点已在 [v4 点集](../../robots/L20_left/l20_r_reference_points_v4.json) 中定义，解剖腕点和指腹接触语义仍待验证。不能用 link 原点或质心替代末端表面点。

厂商 [L20 产品页](https://www.linkerbot.cn/details/LinkerHandL20/)给出 16 主动+5 被动自由度；[ROS SDK](https://github.com/linker-bot/linkerhand-ros-sdk#position-to-finger-joint-mapping-table)的 L20 命令数组有 20 个槽位、4 个 Reserved。命令到关节角的顺序、单位、零点、方向、实际版本与联动准确性仍待核查。不能以数量一致代替接口验证，也不能将 L20 Lite 资料直接套用。

详细实施计划见 [URDF 到虚拟人手 R 指南](../../docs/L20左手_URDF到虚拟人手R建模指南.md)。MANO 环境和 L20 FK 已验证，仿真用初始点已固定；下一步是仅尺度与 β+尺度的几何拟合。结构属性与在线来源审计相符不等于锁定了上游 commit。

## 2026-10-17 补充：用户本地目录复核

重新读取用户目录 `E:\lingqiaoshou\linkerhand-urdf-main\linkerhand-urdf-main\l20\left`。其中的 `linkerhand_l20_left.urdf` 为 1204 行，包含 22 个 link、21 个 revolute joint 和 5 条 mimic；`meshes` 中 22 个不同 STL 均存在。URDF 的 visual/collision 网格原点均为零，未指定 mesh scale；关节原点、轴、限位和惯性参数可用于文件级 FK 与目标虚拟手运动学骨架。

这套资源可以支持共享 `R_L20` 的初始几何/拓扑定义，但还需要明确 MANO 21 点与机械关节代理的语义对应，并单独补充真实驱动 `u` 的命令顺序、单位、零点、方向和柔顺/肌腱关系。此次只读复核没有新增拟合或性能结果。

## 2026-10-16 补充：网格可提供初始局部取点

URDF 中 `hand_base_link` 和五个 `*_distal` 的 visual/collision 网格引用均为零位姿，未指定 `scale`；因此当前文件层面的 STL 顶点就是相应 link 坐标系中的米制几何。本轮已按端盖/安装平面规则选取腕部/掌部参考和五个指尖初始点，记录在 [v4 点集](../../robots/L20_left/l20_r_reference_points_v4.json)；再通过 FK 转到 `hand_base_link` 或世界坐标，用于 MANO-R 的初始几何拟合。

STL 只保存三角形表面，不保存“腕部”“解剖指尖”或“指腹接触中心”等语义标签。v4 的点使用端盖三角面面积加权中心，选择规则写入报告，并要求在 MANO 一侧使用相同语义；包围盒最小/最大值不能自动当作指尖点。真实硬件测量仍用于检查 CAD 与实物外壳、装配零位、接触面和 SDK 映射，当前不作为开始网格取点和 R 拟合的前置条件。v5 已完成一版仿真 MANO-R 静态拟合，结果和残差见 `experiment/v5_MANO-R拟合/fit_report.md`。
