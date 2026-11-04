# 浏览器工作台（tools/web）

本地网页版 DPanel：改参数 → 生成 OCC 实体 → 划分半域面元网格 → 求解 → 看图看数。
它只是现有求解器的入口，**不引入新的气动实现**。

```bash
python -m tools.web.server --port 8811     # 然后开 http://127.0.0.1:8811/
```

## 它做了什么

| 步骤 | 真正的执行者 | 产物（`runs/web/<时间戳>-<hex6>/`） |
| --- | --- | --- |
| 建体 | 子进程 `python -m tools.web.step_builder`（gmsh + OCC 放样） | `wing.step`、`trailing_edge.json`、`build.json` |
| 网格 | 子进程 `python -m panel.cad_half` | `mesh/surface.json`、`mesh/manifest.json`、`mesh.log` |
| 求解 | 进程内 torch：固定尾迹 `p1_solve_torch` / 自由尾迹 `p1_steady_free_wake` | `summary.json`、`job.log` |

两次子进程都是为了绕开本机限制：gmsh 只能在主线程 `initialize()`（HTTP 线程里会抛
`signal only works in main thread`），而且沙箱禁止管道 stdio，所以子进程 stdout 一律写 `*.log` 文件，
父进程再增量转发到 `job.log`。

## 几何契约（三条硬约束，改动前先读）

1. **后缘必须落在 z=0**：`lifting3d/cad_contact.py` 的 `validate_te_polyline` 直接拒绝离开 z=0 的后缘折线，
   所以参数里没有上反角，每个剖面绕自己的后缘点旋转扭转角，后缘恒在 `(te_x, y, 0)`。
2. **放样必须 ruled**：光滑放样的后缘是 B 样条，与声明的折线差约 3e-3，
   `cad_te.validate_declared_te(..., 1e-7)` 会报 `Declared TE does not lie on a trimmed wing boundary curve`。
3. **每个剖面两条 B 样条**（上表面 LE→TE、下表面 TE→LE）：41 点折线放样会产生约 2000 个 OCC 面、
   STEP 导入要 3 分钟以上；两条样条只要 1.3 秒，后缘依然是尖的。

`panel/cad_half.py` 内部按参考弦归一化，所以 `pipeline.mesh_surface` 传进去的是
`size/chord`、`radius = 1.5·reach + 4·size`（reach 取后缘折线的 max|x|,|y|）。

## HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/schema` | 参数表、默认值、参考量 |
| POST | `/api/build` | body `{"params": {...}}`，返回 job 名与显示网格 |
| POST | `/api/solve` | body `{"job": ..., "alpha": 4, "wake": "fixed"/"free", "size": 0.3, "iterations": 3}` |
| POST | `/api/cancel` | 置取消标志，子进程被杀、worker 线程退出 |
| GET | `/api/job?id=...` | 状态 `built/meshing/solving/done/error/cancelled`、日志、结果 |

## 已知问题

- **自由尾迹会塌陷**：典型 CL 从 +0.39 掉到 +0.26，`converged=false`、`geometry_stable=false`。
  这是 `docs/ISSUE92_B3_VSPAERO_PROTOCOL.md` §13 记录的未解决缺陷，工作台会在结果表里打黄色警告，
  数值不可当气动结论。
- **粗网格可能拓扑失败**：`sections=4` + `size=0.6` 出现过
  `ValueError: Non-manifold right wall edge`（`panel/half_surface_mesh.py`），界面会提示加密网格或增加展向站数。
- GPU 不可选：两条路径都只暴露 CPU 设备（torch 版 p1 单翼路径才有 GPU）。
