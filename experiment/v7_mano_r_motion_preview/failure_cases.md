# v7 失败记录

## v7-render-internal-mesh（已废弃）

- **现象：**在 Windows 上运行 MANO-R MuJoCo 动态预览时，Python 进程异常退出，报告代码为 `0xc06d007f`。
- **触发路径：**旧实现直接写入 `MjModel.mesh_vert`，并调用 `numpy.linalg.svd` 估计网格刚体变换。
- **判断：**这是预览脚本的数值库/内部数组更新路径问题，不是 MANO_LEFT.pkl、L20 URDF 或 v6 静态场景加载错误。
- **处理：**保留本记录，停用旧路径；改成每帧写 OBJ 并从 XML 重新编译 MuJoCo 模型，关闭 MANO-R 碰撞。
- **修复验证：**单帧和 9 帧渲染均退出码为 0，结果见当前目录 `rendered/`。
