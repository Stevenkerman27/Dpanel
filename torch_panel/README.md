# torch_panel

从 GRASS（上游 commit `aa6d81f`）提取出来的可微 PyTorch 面元内核，独立于 JAX 求解器，
只依赖 `torch`（无 numba、无 csdl）。

| 文件 | 作用 |
| --- | --- |
| `kernels.py` | 三角形一次/偶极势核与几何梯度（torch 自动微分） |
| `symmetry.py` | 镜像半模装配：`A_half = A_RR + A_RL`、反射法向、绕序 |
| `nonlifting.py` | 非升力面（源分布）势与几何梯度 |
| `__init__.py` | 公开导出 |

许可证：`LICENSE-APACHE-2.0.txt`（沿用上游）。

测试在 `test/test_torch_panel_{kernels,symmetry,nonlifting,validation,vs_numba}.py`，
运行需要隔离 torch 环境（见 `runs/torch_site`）与 `TMP/TEMP/TMPDIR` 指向可写目录
（`core/triangle_kernel.py` 的 `@njit(cache=True)` 在沙箱默认临时目录上会阻塞）。

验收记录：[`docs/TORCH_PANEL_EXTRACTION.md`](../docs/TORCH_PANEL_EXTRACTION.md)（PR #7）。
