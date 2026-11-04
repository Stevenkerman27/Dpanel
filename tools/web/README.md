# 浏览器工作台（tools/web）

本地网页版 DPanel：改参数 → 生成 OCC 实体 → 划分半域面元网格 → 求解 → 看图看数。
它只是现有求解器的入口，**不引入新的气动实现**。

```bash
python -m tools.web.server --port 8811     # 然后开 http://127.0.0.1:8811/
```

## 界面

- 左侧上半是视口，两个标签页：**几何**（白色实体 + 灰色棱线）与**结果**（在求解器网格上按 Cp 着色，
  尾迹片半透明蓝）。半域网格在结果页按 y=0 镜像成完整翼，着色是**逐面板 Cp**，不是插值。
- 左侧下半是结果表（CL/CD/CM、面板数、未知量、残差、耗时、黄色警告）、Γ 展向曲线或自由尾迹 CL 迭代曲线、运行日志。
- 右侧上半是**部件列表**（升力面 + 若干机身，点击选中、删除），下半按选中部件生成参数表单。
- 右侧最下是求解选项（迎角、网格尺寸、固定/自由尾迹、迭代次数）与「生成几何 / 求解 / 取消」。
- 鼠标左键拖动旋转，滚轮缩放。

## 它做了什么

| 步骤 | 真正的执行者 | 产物（`runs/web/<时间戳>-<hex6>/`） |
| --- | --- | --- |
| 建体 | 子进程 `python -m tools.web.step_builder`（gmsh + OCC 放样） | `wing.step`、`body.step`（有机身时）、`trailing_edge.json`、`build.json` |
| 网格 | 子进程 `python -m panel.cad_half` | `mesh/surface.json`、`mesh/manifest.json`、`mesh.log` |
| 求解 | 进程内 torch：固定尾迹 `p1_solve_torch` / 自由尾迹 `p1_steady_free_wake` | `summary.json`、`job.log` |

两次子进程都是为了绕开本机限制：gmsh 只能在主线程 `initialize()`（HTTP 线程里会抛
`signal only works in main thread`），而且沙箱禁止管道 stdio，所以子进程 stdout 一律写 `*.log` 文件，
父进程再增量转发到 `job.log`。

## 部件模型

求解器只有**一套尾迹系统**，所以部件列表强制：

- **升力面恰好一个**（`kind="wing"`）：参数见 `params.SCHEMA`，后缘折线就是 Kutta 条件的作用线。
- **机身任意多个**（`kind="body"`）：椭球，参数 `length / diameter / nose_x`，面心固定在
  `z = -0.75·diameter`（挂在机翼下方、与翼面留出间隙，避免实体相交）。机身壁面以
  `--body-cad` 交给 `panel/cad_half.py`，被归为**非升力壁面**：有速度势与 Cp，但不参与环量/尾迹。

多加一个升力面需要多套尾迹与多条后缘，当前离散做不到（`pipeline.split_parts` 会直接报错）。

## 几何契约（三条硬约束，改动前先读）

1. **后缘必须落在 z=0**：`lifting3d/cad_contact.py` 的 `validate_te_polyline` 直接拒绝离开 z=0 的后缘折线，
   所以参数里没有上反角，每个剖面绕自己的后缘点旋转扭转角，后缘恒在 `(te_x, y, 0)`。
   根因是平坦尾迹模型：`panel/cad_half.py` 的尾迹片是 `z=0` 的平面四边形，无法表示非平面尾迹。
2. **放样必须 ruled**：光滑放样的后缘是 B 样条，与声明的折线差约 3e-3，
   `cad_te.validate_declared_te(..., 1e-7)` 会报 `Declared TE does not lie on a trimmed wing boundary curve`。
3. **每个剖面两条 B 样条**（上表面 LE→TE、下表面 TE→LE）：41 点折线放样会产生约 2000 个 OCC 面、
   STEP 导入要 3 分钟以上；两条样条只要 1.3 秒，后缘依然是尖的。

`panel/cad_half.py` 内部按参考弦归一化，所以 `pipeline.mesh_surface` 传进去的是
`size/chord`、`radius = 1.5·reach + 4·size`（reach 取后缘折线的 max|x|,|y|）。

## HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/schema` | 升力面参数表 `params`、机身参数表 `body_params`、默认值、参考量 |
| POST | `/api/build` | body `{"parts":[{"kind":"wing","params":{...}},{"kind":"body","params":{...}}]}`，返回 job 名与显示网格 |
| POST | `/api/solve` | body `{"job": ..., "alpha": 4, "wake": "fixed"/"free", "size": 0.3, "iterations": 3}` |
| POST | `/api/cancel` | 置取消标志，子进程被杀、worker 线程退出 |
| GET | `/api/job?id=...` | 状态 `built/meshing/solving/done/error/cancelled`、日志、结果 |

旧的裸参数写法 `{"params": {...}}` 仍然可用（等价于单个升力面）。

固定尾迹成功时 `result.field` 是结果页的数据：

```
field = {vertices: [[x,y,z], ...],   # 半域求解器网格（含尾迹节点，物理坐标）
         faces:    [[i,j,k], ...],   # 壁面三角形
         cp:       [Cp, ...],        # 逐壁面面板，长度 == len(faces)
         wake_faces: [[i,j,k], ...]}
```

## 已知问题

- **自由尾迹会塌陷**：典型 CL 从 +0.39 掉到 +0.26，`converged=false`、`geometry_stable=false`。
  这是 `docs/ISSUE92_B3_VSPAERO_PROTOCOL.md` §13 记录的未解决缺陷，工作台会在结果表里打黄色警告，
  数值不可当气动结论。自由尾迹路径不产生 `field`，结果页会回落到几何视图。
- **机身的数值影响尚未验收**：同一尺寸下，机翼-only 为 204 面板 / CL +0.4219 / CD 0.01316，
  加机身变成 276 面板 / CL +0.4045 / CD 0.00933。机身壁面只增加浸润面，理论上 CD 应该上升，
  实测反而下降 29%，说明差值不是纯物理效应（离散或压力积分口径需要查）。unchecked 之前，
  机身只用于几何/流场展示，不能当作带机身的整机气动结论。
- **粗网格可能拓扑失败**：`sections=4` + `size=0.6` 出现过
  `ValueError: Non-manifold right wall edge`（`panel/half_surface_mesh.py`），界面会提示加密网格或增加展向站数。
- GPU 不可选：两条路径都只暴露 CPU 设备（torch 版 p1 单翼路径才有 GPU）。
