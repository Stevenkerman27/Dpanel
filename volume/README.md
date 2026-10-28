# volume

体积网格与自适应求解（P2 四面体 FEM、误差指示子、壁面自适应、CAD 椭球回归）。

| 分组 | 文件 |
| --- | --- |
| 网格 | `build_mesh.py`、`build_curved_mesh.py`、`build_cad_mesh.py`、`compact_mesh.py`、`elevate_mesh.py`、`probe_cad_import.py` |
| FEM | `solve.py`、`p2.py`、`solve_p2.py`、`MESH.md`、`FEM_PROTOCOL.md`、`P2_PROTOCOL.md` |
| 自适应 | `error_indicator.py`、`run_adaptation.py`、`ADAPTATION_PLAN.md`、`ADAPTATION_DELIVERY.md` |
| CAD/回归 | `test_cad_ellipsoid.py`、`test_cad_workflow.py`、`test_error_indicator.py`、`test_fem.py`、`CAD_PROTOCOL.md`、`ELLIPSOID_RESULTS.md` |

产物：`volume/results/`、`volume/meshes/`；`.gitignore` 保留 `results/cad_workflow_v1/sphere/input.step` 作为最小 CAD 夹具。
