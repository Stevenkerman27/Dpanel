# test

后端探针与迁移测试。**不是** `panel/` 的单元测试（那些在 `panel/verify*.py` 与各自的 pytest 文件里）。

| 文件 | 作用 |
| --- | --- |
| `export_vortexad_aic.py` | 在锁定 VortexAD/CSDL 环境导出固定几何 AIC/RHS（float64 NPZ） |
| `probe_linear_solvers.py`、`test_probe_linear_solvers.py` | LU / GMRES / GMRES+Jacobi 正反向基准（issue #3、PR #5） |
| `patch_vortexad_raw_cp.py` | 给 VortexAD 打 raw Cp 补丁（不改上游默认行为） |
| `build_wingbody_panel_case.py`、`run_wingbody_panel_loop.py`、`audit_wingbody_panel_loop.py`、`report_wingbody_panel_loop.py` | 翼身回路算例生成、运行、独立审计与报告 |
| `test_symmetry_diagnostics.py` | 独立对称性诊断（伪配对回归） |
| `test_torch_panel_*.py` | `torch_panel/` 的迁移测试（含对 Dpanel numba 核的交叉校验） |

运行环境：

- VortexAD 相关：`PYTHONPATH` 需包含 `runs/backend_probes/src/VortexAD`，`JAX_PLATFORM_NAME=cpu`；
- numba 相关：必须 `TMP=TEMP=TMPDIR=runs/pipwork`（沙箱默认临时目录会让 `njit(cache=True)` 无限阻塞）；
- torch 相关：`PYTHONPATH=runs/torch_site;<repo>`（torch 2.14.1+cpu）。

结果与报告：`runs/backend_probes/`、[`docs/PR5`](../docs/STAGE1_SOLVER_EFFICIENCY.md)、
[`docs/PR7`](../docs/TORCH_PANEL_EXTRACTION.md)。
