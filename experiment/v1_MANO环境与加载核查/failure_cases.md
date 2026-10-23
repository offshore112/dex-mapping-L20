# v1 失败记录

2026-10-14。环境创建和 `pip check` 成功；模型数值检查及 PowerShell 激活首次失败。保留 `run.log`、`code_snapshot.py` 与 `activation_attempt.ps1`，不覆盖本次记录。

1. 首次中性前向与原始模板的最大绝对差为 6.308303282520633e-9 模型单位，超过计划中的 1e-9 容差。检查安装的 smplx 0.1.28 源码：`utils.to_np` 默认转为 np.float32，`body_models.py` 在构造模板、关节回归器与蒙皮权重等张量前使用这个转换，即使指定 torch.float64 仍保留转换舍入。该失败反映数值检查规则不匹配加载器精度；不是已经测得的机器人拟合误差。
2. 激活报 `Missing expression after unary operator '--'`，随后 Python 仍是 D:\conda\python.exe（3.9.13）。检查 Conda 22.9 模块，发现其调用会传空的 _CE_M/_CE_CONDA；当前 PowerShell 的新式原生参数传递保留空参数。下一版在当前会话中采用 Legacy 参数传递规则，不修改用户 profile 或 base 包。

下一版本：`v2_MANO加载兼容修正`；数值容差改为基于已知单精度转换与模型尺度的 `10*eps32*max_abs(v_template)`，并记录实际误差、形状梯度与姿态扰动。该容差仅用于加载一致性，不作为未来 MANO→L20 几何拟合精度阈值。
