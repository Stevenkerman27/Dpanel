# lifting3d

旧一代三维升力面求解器：OCC 建模 + Gmsh 网格 + 面元/尾迹装配 + 报告。已被 `panel/` 取代，
保留用于对照与方法溯源。

| 分组 | 文件 |
| --- | --- |
| 网格 | `build_mesh.py`、`build_cad_mesh.py`、`cad_basis.py`、`cad_components.py`、`cad_cut.py`、`cad_contact.py`、`cad_te.py` |
| 审计 | `audit_*.py`（planform、root cut、wall banks、wake basis、material jump、mesh motion） |
| 输出 | `analyze.py`、`export_surface.py`（VTK XML） |
| 协议 | `PROTOCOL.md`、`CAD_*.md`、`GEOMETRY_*.md`、`ROOT_*.md`、`WAKE_*.md`、`SCHUR_PROTOCOL.md`、`QUADRATURE_PROTOCOL.md` |
| 结果 | `RESULTS.md`、`WINGBODY_RESULTS.md`、`KUTTA_RESULTS.md`、`LL_RESULTS.md`、`PLANFORM_*.md` |

运行产物在 `lifting3d/results/`（每条协议一套 `results/<topic>_vN/`），网格在 `lifting3d/meshes/`。
本目录的 `results/cad_inputs_v1/naca4412_wing_body/` 是仓库保留的 CAD 测试输入
（`wing.step`、`body.step`、`trailing_edge.json`），`panel/cad_half.py` 的半模验收直接使用它。
