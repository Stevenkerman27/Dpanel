# torch_panel

Dpanel 中独立、可审计的 PyTorch 边界元研究实现（不替换 `panel/solve.py`）。

- `kernels.py` / `symmetry.py` / `nonlifting.py`：来自 GRASS 的解析三角面元和非升力组件，沿用已有测试。
- **`half_lifting.py`**：本 PR 新增，直接输入正半侧翼身公共表面，使用 PyTorch Float64 在 CPU/CUDA 计算源/偶极子、镜像影响、有限尾迹线性展向基函数、弱 Kutta、半模稠密 LU、显式一阶转置伴随、切边感知的原始 Cp 和局部力矩。连续几何张量留在 PyTorch，固定拓扑先由 CAD/主机生成。适用域为严格 XZ 对称、零侧滑、平面尾迹、有效接触拓扑。

从项目根目录使用：

```python
from torch_panel.half_lifting import solve_half_torch, topology_to_torch
```

**数值验收**：`python -m pytest -q test/test_torch_half_lifting.py`；同一保存场的 JAX↔Torch CPU/CUDA 比较见 `python -m panel.verify_torch_half --help`。每个比较报告源 JSON SHA、Float64 真残差、逐字段容差、局部几何导数和真实 GPU allocated/reserved（有 NVIDIA GPU 时）。不要求位一致；小数值差异允许，但不可变更内侧双层自迹 −1/2、真实尾缘/压力切边约定或隐藏压力裁剪。

Torch 半模的 `ImplicitLUSolve` 用 `torch.linalg.lu_solve(..., adjoint=True)` 计算转置伴随；PyTorch 官方文档：[lu_factor](https://docs.pytorch.org/docs/stable/generated/torch.linalg.lu_factor.html)、[lu_solve](https://docs.pytorch.org/docs/stable/generated/torch.linalg.lu_solve.html)。CUDA 上 LU factor 可能有 CPU/GPU 同步，其性能需要实测。原型仅证明可行性，不能把 PyTorch Torch2.x CPU 结果当成 CUDA 验收。

该模块不做 CAD/重网格微分、不实现 GPU GMRES/FMM/H²，也不声称气动物理准确；`physical_accuracy_accepted=false`，正式 GRASS 训练仍暂停。
