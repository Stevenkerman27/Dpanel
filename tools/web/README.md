# 浏览器工作台（tools/web）

本地网页版 DPanel：改参数 → 生成 OCC 实体 → 划分半域面元网格 → 求解 → 看图看数。
它只是现有求解器的入口，**不引入新的气动实现**。

```bash
python -m tools.web.server --port 8811     # 然后开 http://127.0.0.1:8811/
```

## 界面

- 左侧上半是视口，三个标签页：**几何**（参数化显示网格）、**网格**（求解器实际壁面三角网格）与**结果**（在求解器网格上按 Cp 着色，
  尾迹片半透明蓝）。半域网格在结果页按 y=0 镜像成完整翼，着色是**逐面板 Cp**，不是插值。
- 左侧下半是结果表（CL/CD/CM、面板数、未知量、残差、耗时、黄色警告）、Γ 展向曲线或自由尾迹 CL 迭代曲线、运行日志。
- 右侧上半是**部件列表**（升力面 + 若干机身，点击选中、删除），下半按选中部件生成参数表单。
- 右侧最下是求解选项（迎角、网格尺寸、固定/自由尾迹、迭代次数）与「生成几何 / 生成网格 / 求解 / 取消」。网格尺寸以米为单位，尺寸越小通常面板越密；面板数由几何和网格尺寸共同决定，没有直接指定面板总数的输入项。先生成网格，再单独求解；实际壁面/尾迹面板数显示在网格标签页。改变几何、尺寸或迎角后需要重新生成网格。
- 左下角有 xyz 坐标轴（X 红 / Y 绿 / Z 蓝，与视图同一旋转矩阵）。
- 鼠标左键拖动旋转（上下拖动方向反转），滚轮缩放；相机按**壁面几何的世界坐标包围盒**定标（只在换场景时重算），结果视图不把尾迹计入居中与缩放，
  所以任意角度下缩放一致，左右拖动只改偏航、不改俯仰（俯仰夹在 ±85°）。
- 求解选项里有**设备**下拉框：现在两条求解路径都只有 CPU，选 GPU 会明确回退并在结果表标注。

## 它做了什么

| 步骤 | 真正的执行者 | 产物（`runs/web/<时间戳>-<hex6>/`） |
| --- | --- | --- |
| 建体 | 子进程 `python -m tools.web.step_builder`（gmsh + OCC 放样） | `wing.step`、`body.step`（有机身时）、`trailing_edge.json`、`build.json` |
| 网格 | 子进程 `python -m panel.cad_half` | `mesh/surface.json`、`mesh/manifest.json`、`mesh.log` |
| 求解 | 子进程 `python -m tools.web.solve_runner`（jax 半域稠密解 / 自由尾迹） | `summary.json`、`job.log` |

两次子进程都是为了绕开本机限制：gmsh 只能在主线程 `initialize()`（HTTP 线程里会抛
`signal only works in main thread`），而且沙箱禁止管道 stdio，所以子进程 stdout 一律写 `*.log` 文件，
父进程再增量转发到 `job.log`。

## 部件模型

求解器只有**一套尾迹系统**，所以部件列表强制：

- **升力面恰好一个**（`kind="wing"`）：参数见 `params.SCHEMA`，后缘折线就是 Kutta 条件的作用线。
- **机身任意多个**（`kind="body"`）：椭球，参数 `length / diameter / center_x / center_z`
  （绝对位置，y 恒为 0，因为半域以 y=0 为对称面）。机身壁面以 `--body-cad` 交给
  `panel/cad_half.py`，被归为**非升力壁面**：有速度势与 Cp，但不参与环量/尾迹。
- 升力面的 `origin_x` 是**根前缘的绝对 x**；没有 y 偏移（对称面）。z 由 `dihedral_deg` 给出：
  每个剖面绕自己的后缘点滚转，后缘高度 `te_z = |y|·tan(dihedral)`，根部后缘恒在 z=0。
  所以后缘是一条**任意三维折线**（上反角把展向抬起来），不再限制在 z=0。

多加一个升力面需要多套尾迹与多条后缘，当前离散做不到（`pipeline.split_parts` 会直接报错）。

## 几何契约（三条硬约束，改动前先读）

1. **后缘是任意三维折线，尾迹是它沿 x 拉伸出的 +z 图面**：`validate_te_polyline` 只要求点有限、
   相邻点相异、y 严格递增，不再要求 z=0（`z=0` 门在 3-D 后缘那次改动里删掉了）。
   `panel/cad_half.py` 的尾迹条四角取 `np.interp(y, te[:,1], te[:,2])`，所以条内 z 沿 y 线性、
   四角共面；尾迹面法向的 z 分量必须为正（`+z graph`）。上反角就是靠这条走的。
   求解层原本有 5 处"水平尾迹"断言（`half_surface_mesh` / `half_native` / `shed_groups`），
   已放宽为"必须是 +z 图面"；核函数 `panel/kernel.py` 的线性环量项也补了 `n_y` 修正
   （`offset_y = y_t − y_c − n_y·h`），否则倾斜尾迹的线性项在 20° 时比真值大 13 倍。
2. **放样必须 ruled**：光滑放样的后缘是 B 样条，与声明的折线差约 3e-3，
   `cad_te.validate_declared_te(..., 1e-7)` 会报 `Declared TE does not lie on a trimmed wing boundary curve`。
3. **每个剖面两条 B 样条**（上表面 LE→TE、下表面 TE→LE）：41 点折线放样会产生约 2000 个 OCC 面、
   STEP 导入要 3 分钟以上；两条样条只要 1.3 秒，后缘依然是尖的。

`panel/cad_half.py` 内部按参考弦归一化，所以 `pipeline.mesh_surface` 传进去的是
`size/chord`、`radius = 1.5·reach + 4·size`（reach 取后缘折线的 max|x|,|y|）。

## 上反角（`dihedral_deg`，±20°）

几何层是整剖面绕自己的后缘点滚转（`tools/web/geometry.py:_placement`），网格与尾迹一起抬起来。
实测（`runs/scratch/dih_case.py`，size 0.3、alpha 4、固定尾迹、`--ncpu 1`）：

| 上反角 | 后缘 z | 尾迹 z | 面板 | CL | CD | CM_y |
| --- | --- | --- | --- | --- | --- | --- |
| 0° | 0.0000 | 0.0000 | 228 | +0.4279 | 0.00950 | −0.1360 |
| 5° | 0.2625 | 0.2625 | 236 | +0.4570 | 0.01024 | −0.1505 |
| 10° | 0.5290 | 0.5290 | 238 | +0.4611 | 0.01203 | −0.1560 |
| 20° | 1.0919 | 1.0919 | 254 | +0.4392 | 0.01019 | −0.1509 |

"尾迹 z" 是求解器网格里尾迹节点的 max|z|，恒等于声明后缘的 max|z|，说明尾迹真的跟着弯后缘走。
0° 与改动前的代码用同一份 `surface.json` 复算，CL/CD 差 1.4e-16（平面路径没变）。

**这些数只是"能算、能收敛"，不是气动结论**：投影展长固定后，滚转使实际翼面积与物理展长各增
`1/cosθ`，与升力方向的 `cosθ` 投影相抵，所以 CL 近似不变（20° 时 +2.6%，与这个机制一致）；
但 5°/10° 的非单调凸起（+7%）和 CD 的变化没有验收依据。

**复现必须 `--ncpu 1`**：`panel/cad_half` 在 `--ncpu 8` 下同一输入会给出不同坐标
（maxdiff 0.389，面板数相同），0° 的 CL 会在 0.4173–0.4350 间跳；`--ncpu 1` 两次结果逐位相同。

## HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/schema` | 升力面参数表 `params`、机身参数表 `body_params`、默认值、参考量 |
| POST | `/api/build` | body `{"parts":[{"kind":"wing","params":{...}},{"kind":"body","params":{...}}]}`，返回 job 名与显示网格 |
| POST | `/api/mesh` | body `{"job": ..., "alpha": 4, "size": 0.3}`；生成并缓存实际壁面/尾迹网格 |
| POST | `/api/solve` | body `{"job": ..., "alpha": 4, "wake": "fixed"/"free", "size": 0.3, "iterations": 3}`；只使用匹配迎角和尺寸的已生成网格 |
| GET | `/api/mesh?id=...` | 返回已生成网格的坐标、壁面三角面与尾迹面板数 |
| POST | `/api/cancel` | 置取消标志，子进程被杀、worker 线程退出 |
| GET | `/api/job?id=...` | 状态 `built/meshing/meshed/solving/done/error/cancelled`、日志、结果 |

旧的裸参数写法 `{"params": {...}}` 仍然可用（等价于单个升力面）。

固定尾迹成功时 `result.field` 是结果页的数据：

```
field = {vertices: [[x,y,z], ...],   # 半域求解器网格（含尾迹节点，物理坐标）
         faces:    [[i,j,k], ...],   # 壁面三角形
         cp:       [Cp, ...],        # 逐壁面面板，长度 == len(faces)
         wake_faces: [[i,j,k], ...]}
```

## 已知问题

- **上反角还没有物理验收**：见上一节的免责说明；5°/10° 的 CL 凸起未解释。自由尾迹 + 20° 上反角
  能跑但照样塌陷（CL 0.439 → 0.309 → 0.240，两轮迭代），和平面一致。
- **自由尾迹会塌陷**：典型 CL 从 +0.39 掉到 +0.26，`converged=false`、`geometry_stable=false`。
  这是 `docs/ISSUE92_B3_VSPAERO_PROTOCOL.md` §13 记录的未解决缺陷，工作台会在结果表里打黄色警告，
  数值不可当气动结论。自由尾迹路径不产生 `field`，结果页会回落到几何视图。
- **机身的数值影响尚未验收**：同一尺寸下，机翼-only 为 204 面板 / CL +0.4219 / CD 0.01316，
  加机身变成 276 面板 / CL +0.4045 / CD 0.00933。机身壁面只增加浸润面，理论上 CD 应该上升，
  实测反而下降 29%，说明差值不是纯物理效应（离散或压力积分口径需要查）。unchecked 之前，
  机身只用于几何/流场展示，不能当作带机身的整机气动结论。
- **粗网格可能拓扑失败**：`sections=4` + `size=0.6` 出现过
  `ValueError: Non-manifold right wall edge`（`panel/half_surface_mesh.py`），界面会提示加密网格或增加展向站数。
- gpu 不可选：两条路径都只暴露 CPU 设备（torch 版 p1 单翼路径才有 GPU）。
- **求解进程会偶发崩溃（0xC0000005）**：本机 jax（0.12.0.1, x64）在编译/执行约 300 未知量的
  稠密解时约一半概率直接杀死解释器，与网格和部件位置无关（同一 surface 连续重跑，成功/崩溃随机出现）。
  `panel/differentiable.py` 的 `linear_solve` 已从 host LAPACK（`jla.lu_factor`）改成
  `jnp.linalg.solve`（XLA 的 LU），崩溃率下降但没消失；结果数值与原路径逐位一致。
  因此求解放在子进程里跑，崩了由 `pipeline.solve` 自动重试（最多 6 次，通常 1–2 次内成功），
  服务器的进程始终不受影响。连续 6 次失败才会把带退出码的错误写进结果表。
