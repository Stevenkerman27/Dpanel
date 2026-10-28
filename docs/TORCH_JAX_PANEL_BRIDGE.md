# PyTorch—JAX 面元载荷桥：第一阶段 CPU 验收

状态：基于 PR #13 `9d68446` 的隔离接口实现；不改面元方程、压力恢复、CAD、原 CLI 或 GRASS 默认训练路径。对应 Issue #14。合并由本地验收后处理。

## 已有证据和本次边界

PR #13 本地普通评论 `6396019564`：五个真实构型（484/696/616/1646/1976 壁面）均通过旧场回归及任意局部载荷几何 VJP；六项单元测试通过。Cp 差约 2–3e-14，面力相对 L2 约 1e-15。该结果只证明离散接口/导数一致，不证明物理载荷准确。压力恢复误差尚未解决，`physical_accuracy_accepted=false` 保持不变。

本次目标：让 PyTorch 内的连续几何参数或网络权重，经过这个 JAX 求解器产生损失后，可以得到正确的一阶梯度。**不把几何导数的 PyTorch 包装等同于 GRASS 的完整几何生成链交付。**

## 接口

```python
import jax
import torch
from panel.torch_bridge import TorchPanelBridge

jax.config.update("jax_enable_x64", True)  # caller decides; module does not change globals
bridge = TorchPanelBridge(topology)       # construct once per valid topology
# All continuous arguments are CPU torch.float64 tensors; scalars have shape ().
loads = bridge(vertices, station_y, freestream, density,
               reference_area, reference_chord, reference_point)
loss = (loads.panel_forces * force_seed).sum()
loss = loss + (loads.panel_moments * moment_seed).sum()
loss.backward()
```

七组连续输入保持与 PR #13 `solve_fields` 相同语义：全部物面和尾迹节点、当前环量站位、来流向量、密度、Sref、Cref 和力矩原点。输出为 `Cp_raw`、`panel_forces`、`panel_moments`、`CL`、`CD_pressure`、`CM`。SI 输入下力为 N，矩为 N·m；CG 或参考量是设计变量时可以参与反向传播。部件汇总可由固定标签在 Torch 中对面力/面矩求和，不另造气动力。

## 反向机制和内存口径

- `torch.autograd.Function` 的 forward 对输入做独立 CPU 拷贝，再调用原 JAX 求解器；detach 只发生在自定义 Function 边界，其导数由 backward 显式提供。
- backward 将六种输出的上游权重作为**一次组合 VJP 的种子**，返回七组输入梯度。没有显式构造完整载荷—节点 Jacobian，没有有限差分 backward；多个独立目标需要多个 VJP，不能把种子数量描述为免费。
- 使用 `save_for_backward` 保存连续 Torch 输入以保持就地修改版本检查；未使用输出的 `None` 梯度按零种子处理。
- 不把 JAX pullback 闭包保存到 Torch ctx；backward 从保存输入重算 VJP，因而会重复状态求解/部分装配。它避免长期持有每次调用的稠密反向缓存，但**尚未测得大型翼身的峰值内存改善或速度优势**，JAX/XLA 自身缓存和反向峰值仍可能是 O(N²)。
- 当前密集求解仍复用 PR #13 的 LU/显式转置隐式求导。未引入 GMRES、FMM 或任何改变压力结果的算法。

这遵循 PyTorch 官方 `Extending autograd` 的外部库 Function 模式（`save_for_backward`、`set_materialize_grads(False)`、`once_differentiable`），及 JAX 官方 `jax.vjp` 与 `custom_linear_solve` 契约：
https://docs.pytorch.org/docs/stable/notes/extending.html
https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html
https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html

## 明确拒绝/尚未实现

只支持 CPU float64 和一阶 reverse-mode，显式拒绝错误精度、形状、零速度/非法参考量、非平面尾迹、条带失配与退化压力模板；不默默转精度、投影几何或裁剪梯度。每次前向使用 PR #13 的局部几何检查，不能替代完整外露表面/接触相交审计。

二阶反向、torch.func/vmap、torch.compile、CUDA/DLPack 零拷贝、batch 批量吞吐、重新网格和接触拓扑变化均不在范围。当前不修改 GRASS 的 CST/机身编码器，不启动正式训练。

## 本次云端实际执行

环境：Python 3.13.5、Torch 2.10.0+cpu、JAX 0.9.0.1、NumPy 2.3.5、SciPy 1.17.0、CPU float64。

`python -m pytest -q test/test_torch_panel_bridge.py --disable-warnings`：**8 passed / 24.96 s**。使用 PR #13 原有 40 壁面、8 尾迹、43 状态未知量的封闭厚翼测试夹具。验证：原生 JAX 前向与七输入 VJP、五参数 Torch 几何映射的 gradcheck/三档方向差分、梯度传回小型 `nn.Linear` 权重、未使用输出和非连续输入、两个未完成反向的调用互不串扰、重复 backward、就地修改检查、二阶反向和非法输入拒绝。

实际执行模块 `panel/differentiable.py`、`panel/kernel.py` 与测试夹具的 git blob SHA 已分别核对为 `173eef18770f7e9e080b9f0849f81a79e8d4c5fe`、`4a6ccaf8c44ee37a7f46c82ffef0b920197dff2c`、`0f9904ffb5a809b9091de3db3788baaceaf8c254`，与 PR #13 一致。此次没有重跑全部旧 CLI 测试。

另以该小翼生成一个**合成保存场输入**测试 CLI 读入链，六输出与原生 JAX 差为零，两方向/三步长检查全部通过；h=1e-3/1e-4 的相对误差约 6.7e-10 至 4.2e-9，h=1e-5 最坏约 2.52e-7。这个 smoke 不是用户的真实翼身保存场，也不用于外推整机速度。完整真实构型验收仍由本地执行。

## 本地下一轮

使用能同时导入已有 Torch 与 JAX 的隔离环境，不改原研究环境、依赖文件或 CI；缺依赖导致 skip 不能算通过。

```bash
python -m pytest -q test/test_torch_panel_bridge.py
python -m panel.verify_torch_bridge \
  --run runs/EXISTING_CASE/solution \
  --out runs/NEW_TORCH_BRIDGE --derivatives
```

`--run` 指向真实含 `surface_input.json/result.json/fields.npz` 的求解目录。默认最多1000壁面；先单翼484、前置机身696、粗双机身616，后审查内存预算才显式提高 `--max-panels`。不再建网格、不覆盖旧目录。`--out` 已存在立即拒绝。报告含字段差、七参数 Torch 形变/工况链的两个方向×三步长 FD、冷/热前向和反向重算时间。峰值RSS需本地父进程另测，脚本不以矩阵字节数冒充RSS。

回传精确分支SHA、Python/Torch/JAX版本、pytest结果与退出码、真实案例字段差/梯度误差/时延/峰值RSS和首个traceback。通过后再另开 GRASS 真实连续参数映射与接口验收；CPU桥可验证链路，但不是面向正式训练吞吐量的最终方案。
