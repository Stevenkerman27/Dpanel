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
- 固定和自由尾迹默认使用 P1 Galerkin 弱壁面形式；“P1 配点”可用于快速试算和对照。
  固定尾迹使用镜像的全翼 P1 网格，比原半域 P0 求解更耗内存与时间；
  结果不应与原 P0 模型当作同一种离散化直接比较。
  原 P0 半域固定尾迹仍可通过 `python -m tools.web.solve_runner --surface <surface.json> --out <new-directory> --wake fixed --wall-formulation p0` 使用。
- **CPU 线程**默认 `min(8, 逻辑 CPU 数)`，可在 1 到逻辑 CPU 数之间调整；只控制求解子进程，
  不改变网格阶段的线程设置。求解日志按校验、装配、求解/载荷、结果检查分阶段显示。
  “求解耗时”显示整个求解子进程的耗时，包含这些阶段，不再只显示部分计算时间。

## 它做了什么

| 步骤 | 真正的执行者 | 产物（`runs/web/<时间戳>-<hex6>/`） |
| --- | --- | --- |
| 建体 | 子进程 `python -m tools.web.step_builder`（gmsh + OCC 放样） | `wing.step`、`body.step`（有机身时）、`trailing_edge.json`、`build.json` |
| 网格 | 子进程 `python -m panel.cad_half` | `mesh/surface.json`、`mesh/manifest.json`、`mesh.log` |
| 求解 | 子进程 `python -m tools.web.solve_runner`（镜像全翼 P1 弱形式 / 配点对照） | `summary.json`、`job.log` |

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

## 扭转翼的后缘分岸

2026-11-05 用户的根/梢扭转 `6/-1°`、2% 弯度输入触发
`TE cut must have distinct upper and lower banks`。原因是旧导出器用壁面重心相对
后缘的 z 正负判上下岸；扭转和弯度可以让两岸重心同在后缘上方，
这不等于两岸合并或网格不共形。

`panel/half_surface_mesh.py` 现在先保持原外向壁面绕向，再按后缘边的向外展向
方向区分同向/反向两岸；仍要求两壁面共享同一条真实 Gmsh 边、绕向相反，
并保留非流形、对称面、后缘声明和尾迹方向检查。没有改接触容差或移动后缘。
该 STEP 在 size 0.1 m、单网格线程下的 32 条 TE 边有 11 条被旧高度判据误拒绝；
修正后完整 CLI 建网通过，932 壁面/538 尾迹面片，导出与冻结原始 Gmsh 数组重放逐项相同。
后续固定尾迹求解实测 946 未知量、CL `0.665389904774`、
线性相对残差 `1.59e-15`，不代表气动精度验收。
单线程全量回归实测 `11 failed, 814 passed, 21 skipped`（352.93 s），
失败 ID 集合与既有基线相同；完整日志哈希保存在证据收据中。

可移植冻结输入、源文件哈希及重放命令在
[TE 分岸证据](../../verification/te_bank_classification/README.md)。
原厚翼对照的完整导出字典与旧版完全一致；新增正负扭转的分岸回归。
Web 下一次生成网格会启动新导出进程，无需重启服务。

## HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/schema` | 升力面参数表 `params`、机身参数表 `body_params`、默认值、参考量 |
| POST | `/api/build` | body `{"parts":[{"kind":"wing","params":{...}},{"kind":"body","params":{...}}]}`，返回 job 名与显示网格 |
| POST | `/api/mesh` | body `{"job": ..., "alpha": 4, "size": 0.3}`；生成并缓存实际壁面/尾迹网格 |
| POST | `/api/solve` | body `{"job": ..., "alpha": 4, "wake": "fixed"/"free", "size": 0.3, "iterations": 3, "ncpu": 8}`；只使用匹配迎角和尺寸的已生成网格，线程数越界返回 HTTP 400 |
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
- **大矩阵触发 OpenBLAS 栈缓冲区越界（0xC0000005）**：2026-11-05 已定位到本机
  OpenBLAS 0.3.34 的 Ryzen 自动调参和 Windows Haswell 内核之间的容量冲突，详见下一节。
  自动重试只隔离崩溃，不能修复此缺陷。`jnp.linalg.solve` 的 CPU 降低结果仍包含
  `lapack_dgetrf_ffi`，此前“换成 XLA LU 即可绕开 LAPACK”的解释不成立。

## 固定尾迹求解性能（2026-11-05）

OpenBLAS 热修复后访问冲突消失，但约 4600 面板的完整求解仍需一分钟。`cProfile` 对原
4598 壁面网格记录：总计 88.39 秒，其中 `validate_shed_groups` 74.11 秒（约 84%），
`_triangle_aabb_intersect` 调用 7,025,342 次；矩阵装配两次，原入口的“求解耗时”还漏掉了
前面的校验、第一次装配和后面的镜像检查。**剖析器有开销，这个 88.39 秒不用于加速比。**

优化保持原来的气动与校验路径：

- 预计算壁面三角形的包围盒，每个尾迹三角形用 NumPy 批量筛选候选，再按原面板顺序
  执行原来的精确检查。容差、后缘岸/共享节点排除、相交判定和预算限制保留。
  仍为 O(尾迹面板数 × 壁面面板数) 的 broad phase，没有裁剪气动影响矩阵。
- 研究入口将已经装配的矩阵传给 `solve_half_groups(..., assembled=...)`，避免重复装配。
  没有跨几何缓存矩阵；库 API 未提供 `assembled` 时仍自行装配，调用者只能传当前输入的装配。
- 求解子进程默认 8 线程（受逻辑 CPU 数限制），可用界面/API 的 `ncpu` 或 CLI `--ncpu`
  调整。线程环境变量在 NumPy/JAX/torch 导入前设置；几何/网格子进程维持原设置。
- 镜像、质量矩阵、Kutta、残差等检查全部保留；记录同步后的分阶段时间。

原 4598 壁面/4612 未知量网格：优化后单线程 11.79 秒、八线程 10.16 秒。
用户实际的 4600 壁面/4614 未知量网格，同一份 `surface.json` 的正常子进程记录从
**62.761 秒降至 9.675 秒，单次前后对照约 6.49 倍**；不是多次中位数基准，也不是与
上述 cProfile 时间相除。八线程新结果的阶段时间为校验 0.205、装配 4.594、
求解/载荷 1.243、检查/报告 3.199 秒，另有模块导入等开销。
与该网格旧结果比较：CL 差 -3.33e-16、CD 差 -1.73e-17、Cp 最大差 2.23e-14、
Γ 最大差 5.00e-16，相对线性残差 2.60e-15。CPU 占用不要求所有阶段满载，
线程数增加也不保证总耗时单调下降。

可复现命令：

```powershell
python -m tools.web.solve_runner --surface runs/web/20261105-153101-076a86/mesh/surface.json --out runs/verification/web_performance/user_mesh_threads8 --wake fixed --ncpu 8
python -m tools.verify_shed_group_optimization --baseline 5ca04e8 --surface runs/web/20261105-141850-a135fd/mesh/surface.json --out runs/verification/web_performance/equivalence.json
python -m pytest -q test/test_shed_group_blocks.py test/test_shed_groups_contract.py test/test_shed_group_mass_identity.py test/test_run_half_groups_entry.py test/test_web_solve_threads.py -p no:cacheprovider
```

第二条命令只从 Git 读取旧源码到内存，不建立第二份工作树。原始大网格及穿壁、接触、
容差边界、预算耗尽等共八种输入，新旧接受/拒绝、审计计数和精确检查调用顺序完全一致；
该命令中原始网格校验 49.967 → 0.195 秒。小翼 0°、10° 的 x/y/z 几何缩放梯度与
旧实现最大差为 0，梯度均有限；这是对旧实现的敏感性等价检查，不是对独立有限差分或
物理真值的验收。筛选本来就在 NumPy 主机端几何校验中，不进入 JAX 的连续求导图。

**既有判据限制单独保留：**上述额外构造的窄穿壁三角形在旧版和新版均被接受。
原精确检查遍历中 `np.roll(tri, -1)` 没有指定 `axis=0`，会按展平数组滚动，不能
视作已经验收的完整三角形相交算法。本轮保留它来保证优化前后判定一致，后续需独立修正
并补几何判据验收；不能因为新旧结果一致，就宣称所有穿壁输入都会被拒绝。

针对性测试 `47 passed in 9.47s`；完整 `python -m pytest -q test -p no:cacheprovider`
（`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`，不设置 CORETYPE）
实测 `11 failed, 807 passed, 21 skipped`，181.49 秒，失败集合与热修复后基线相同，
无访问冲突。浏览器实测 build → mesh → solve，204 壁面，指定 2 线程实际传到结果；
非法线程 HTTP 400、日志阶段、显示耗时、桌面输入框布局及无 JS 异常均验证。
证据在 `runs/verification/web_performance/`：`baseline.pstats`、`equivalence.json`、
`user_mesh_comparison.json`、`targeted_tests_v3.log`、`full_tests.log`、
`browser_smoke.json`、`desktop.png`。初期测试夹具失败日志也保留。

## 大网格崩溃的根因与复现（2026-11-05）

本节记录本机实测结论。生产求解路径和默认环境变量未改；根因诊断后已安装本地运行库
热修复，安装与验收见下方“已安装的运行库热修复”。

### 原始故障

作业 `runs/web/20261105-141850-a135fd`：默认机翼、size=0.1 m、alpha=4°，
4598 壁面 + 1576 尾迹面板，线性系统 4612 阶。Windows 应用程序错误记录在
14:20:18、14:21:25 均报告 `openblas.dll + 0xaaa42e`；UI 在等待进程退出后于
14:20:23、14:21:28 显示重试。这与用户提供的 14:20:24 / 14:21:28 日志吻合。

实测环境：Windows x64，AMD Ryzen 9 9950X，Python 3.13.14，NumPy 2.4.6，
SciPy 1.18.0，JAX/jaxlib 0.12.0.1；当前单一解释器为
`C:/Users/zyx20/anaconda3/envs/myml/python.exe`。出错 DLL 属于 conda-forge
`libopenblas 0.3.34 pthreads_h877e47f_0`，位于该环境的 `Library/bin/openblas.dll`，
SHA-256 为 `61333f15528c05e0684e1efcdaebc44b4cceb3cae5fcd265f9370a654715289c`。
NumPy wheel 另外携带 OpenBLAS 0.3.31，不能只看 `numpy.__version__` 判断故障 DLL。

### 为什么面板数增大才触发

1. 壁面/环量未知量变多，CPU LAPACK 的 LU 分解开始使用较大的分块矩阵乘法。
2. OpenBLAS v0.3.34 的 `kernel/setparam-ref.c:init_parameter` 检测到 AMD、
   L2=1024 KB、L3 为 32768 KB 的倍数且支持 AVX512 时，将 `dgemm_p/q` 改成 512/512。
   这台机器实测运行时表为 `P=512,Q=512,R=32240`。
3. 同版本 `param.h` 原本把 Windows Haswell 的 Q 设为 128。
   `kernel/x86_64/dgemm_kernel_4x8_haswell.S` 的 Windows 分支也据此只预留
   `128*8*12+512 = 12800` 字节的局部缓冲空间，再加 128 字节头部和页对齐余量。
   Ryzen 的运行时调参覆盖了 Q，却没有扩大汇编内核的缓冲区。
   DLL 反汇编也确认：初始化函数先向 P/Q 写入 512/128，随后在 AVX512 条件分支
   改写为 512/512；这不是仅根据源码推测运行时行为。
4. 内核在至少 24 列的路径中，每个 K 向栈缓冲复制 12 个 double，即每轮 96 字节。
   K=128 需 12288 字节；K=256 需 24576 字节；K=512 需 49152 字节。
   后两者超过预留空间，即使算上最多 4095 字节的对齐余量也放不下。
5. 写越界先破坏保存寄存器和返回地址，继续写则碰到线程栈的高地址边界。
   因此症状可表现为内核写内存失败、函数返回时跳到零地址，或在别的 Python/JAX 操作处才退出；
   Python 线程栈不一定就是原始故障点。小算例能过也不能证明运行库安全。

反汇编确认原始地址是 `dgemm_kernel_HASWELL + 0x16e`：
`vmovups [rsi], ymm1`。实际 4612 阶机翼复现的首次异常落在同一复制循环的第三条写指令
（`+0x177`），当时 `K=256`、还剩 2 轮、已推进 24384 字节，异常写地址恰好等于
线程 `StackBase`。1024 阶独立矩阵乘法则捕获到 `K=512`，同样写到 `StackBase`。
这确定了实际的栈越界，不能解释为矩阵病态、几何拓扑失败或普通内存不足。

### 对照证据

所有对照均单线程、独立进程，不导入 torch，不加载网格。

| 输入 | Haswell 原始 Q=512 | 同一 Haswell，仅诊断进程改 Q=128 |
| --- | --- | --- |
| 1024 阶单位矩阵 DGEMM | 3/3 访问冲突 | 3/3 正确 |
| 4612 阶单位矩阵直接 `dgetrf_` | 3/3 访问冲突 | 3/3 正确 |
| 4612 阶单位矩阵 JAX solve | 3/3 访问冲突 | 3/3 正确 |

直接调用同一微内核，4×24×128 的全 1 输入连续 3 次正确；K=192 已捕获保存寄存器被
`double(1)` 的位模式 `0x3ff0000000000000` 覆盖，K=256 捕获越界写至 `StackBase`。
直接 LU 的 128/256/272/320 阶探针此次完成，384 阶返回地址被覆盖后跳到 0，
512/1024/4612 阶捕获缓冲区越界。因此没有一个与面板数一一对应的通用安全阈值；
LU 的内部块形状、调用栈深度和页对齐位置都参与决定何时显现。

Sandybridge 作为另一项诊断对照：三种单位矩阵测试也各 3/3 正确，原始机翼网格完成
（CL=0.5914475754705966、相对线性残差 2.83e-15、约 68.4 秒）。
这只验证绕行与代数求解，不授予物理精度验收。按用户要求，自动绕行改动已撤回；
后续修复处理了调参值与内核容量的契约，独立验收记录见下方。

### 复现命令与证据位置

```powershell
# 以下 python 均指项目规定的 myml 解释器。
python -m tools.blas_probe --case gemm --size 1024 --out runs/verification/native_blas/gemm
python -m tools.blas_probe --case lu --size 4612 --out runs/verification/native_blas/lu
python -m tools.blas_probe --case jax --size 4612 --out runs/verification/native_blas/jax
# 同一 Haswell 内核的因果对照；只改诊断子进程中的 Q，不修改 DLL 文件。
python -m tools.blas_probe --case lu --size 4612 --cores Haswell --gemm-q 128 --out runs/verification/native_blas/lu_q128
# 捕获首次原生异常、寄存器及线程栈边界；不需要安装调试器。
python -m tools.blas_debug --out runs/verification/native_blas/debug_lu4612.json -- -m tools.blas_probe --worker --case lu --size 4612 --out runs/scratch/blas
```

`tools.blas_probe --gemm-q` 的表偏移写操作受上述 DLL 的 SHA-256 限制，只用于因果实验。
默认测试与因果对照分别保留在 `runs/verification/native_blas/{gemm,lu,jax}*/report.json`；
原始 Windows 错误和浏览器日志为同目录下的 `original_windows_events.json`、
`original_web_job.json`，首次异常为 `debug_web.json`、`debug_gemm.json`、
`debug_lu*.json`、`debug_kernel*.json`。原始 surface 只引用一次，不复制。
上游源文件的本机诊断下载位于 `runs/scratch/openblas_setparam.c`、
`openblas_param.h`、`dgemm_kernel_4x8_haswell.S`，版本均为 v0.3.34。

### 正式修复方向与分块性能

2026-11-05 读取 OpenBLAS 上游 `develop`（`02f585444a0fba2505cfda77cc6684de008ea477`）
确认，`kernel/setparam-ref.c` 已用 `#ifndef NO_AVX512` 包住 Ryzen 调参，并只允许
Cooperlake / SkylakeX / SapphireRapids 三类内核应用 Q=512。当前 conda 构建使用
`NO_AVX512=1`，这个修正会保留 Windows Haswell 原有的 Q=128。
发布构建应包含此修正并经过本机验证。2026-11-05 查询 conda-forge 包列表，最新的
Windows pthreads 包仍为 `0.3.34 pthreads_h952cde7_2`；直接加载缓存中的该构建，
实测仍是 `P=512,Q=512,R=32240`，不能把同版本 build 2 当成已修复版本。
本机采用下面明确标记的二进制热修复；诊断工具的内存表改写仍只用于实验。

Q 控制 K 方向分块，影响缓存复用、打包次数和微内核调用开销；Q 减半不意味着浮点运算量
翻倍或速度减半。可接受范围先受微内核缓冲容量限制，其内再按矩阵尺寸和硬件测性能。

单线程、同种随机矩阵（4612×4612，固定种子 `20261105`，元素为标准正态除以 sqrt(N)，
对角加 2），每种配置独立启动三次，**只计 `dgetrf_` 本身**：

| 配置 | LU 中位耗时 | 三次制造解检查 |
| --- | --- | --- |
| Haswell，诊断进程设 Q=128 | 0.8195 s | 3/3 正确 |
| Haswell，诊断进程设 Q=64 | 0.8385 s | 3/3 正确 |
| Sandybridge，原始调参 | 0.9981 s | 3/3 正确 |

制造解为全 1 向量，求解后最大绝对误差要求小于 1e-10。这里 Haswell Q=128 的耗时
比 Q=64 少约 2.3%，比 Sandybridge 少约 17.9%；不能据此推断与已崩溃的 Haswell
Q=512 相比损失多少性能，也不代表实际 AIC 或完整 Web 求解的耗时比例。
该计时不含矩阵生成、分解后的解算/校验、网格、影响矩阵装配或 JAX 编译。

```powershell
python -m tools.blas_probe --case lu --size 4612 --matrix random --cores Haswell --gemm-q 128 --out runs/verification/native_blas/performance_haswell128
python -m tools.blas_probe --case lu --size 4612 --matrix random --cores Haswell --gemm-q 64 --out runs/verification/native_blas/performance_haswell64
python -m tools.blas_probe --case lu --size 4612 --matrix random --cores Sandybridge --out runs/verification/native_blas/performance_sandybridge
```

上述目录的 `*.log` 保留 `operation_seconds` 和正确性检查，`report.json` 保留
配置与每次进程退出码。此前单位矩阵探针的总计时包含构造/校验，不用于这张性能表。

### 已安装的运行库热修复（2026-11-05）

当前 myml 的 `Library/bin/openblas.dll` 已安装**本地二进制热修复，不是新版官方包或
本机重新编译的构建**。本机没有 MSVC/SDK 原生构建工具链；没有用另一种编译器重建整库，
以免同时改变 Fortran/复数 ABI 和依赖。

`tools.blas_runtime_fix` 只接受上文 build 0 的完整 SHA-256。它在 11 个已编译的内核
初始化函数中，将 AVX512 条件前的 `JE +0x44` 改为 `JMP +0x44`，共改 11 字节：
跳过四种精度的 Ryzen P/Q 覆盖，效果对应于当前构建 `NO_AVX512=1` 下上游新增的
`#ifndef NO_AVX512`。微内核、导出符号和 ABI 不变。新进程自动检测 Haswell，实测
`P=512,Q=128,R=130544`；R 由剩余初始化代码重新计算，因此比诊断中仅写 Q 更完整。

修复后 SHA-256：
`e0d23e1c32e26f05719d98d92c9aa26a303a83ba0b66463099677459e48223e8`。
安装使用临时文件加 `os.replace`，不会原地改写 conda 的缓存硬链接；缓存中的原始 DLL
实测仍保留原始哈希。若 Windows 有进程映射着旧 DLL，安装会明确失败，需要先结束该进程。
本次安装没有遇到文件占用，无需重启现有服务器；后续求解子进程会加载新 DLL。

复现顺序（均使用项目规定的 myml Python；无需 `OPENBLAS_CORETYPE` 或 `--gemm-q`）：

```powershell
python -m tools.blas_runtime_fix --out runs/verification/native_blas/runtime_fix
python -m tools.blas_probe --dll runs/verification/native_blas/runtime_fix/openblas.dll --case gemm --size 1024 --cores auto Haswell Sandybridge --out runs/verification/native_blas/candidate_gemm
python -m tools.blas_probe --dll runs/verification/native_blas/runtime_fix/openblas.dll --case lu --size 4612 --matrix random --cores auto --out runs/verification/native_blas/candidate_lu
python -m tools.blas_runtime_fix --out runs/verification/native_blas/runtime_fix --install
python -m tools.blas_verify_runtime --out runs/verification/native_blas/fixed_scipy.json
python -m tools.blas_probe --case lu --size 8192 --matrix random --cores auto --out runs/verification/native_blas/fixed_lu8192
python -m tools.blas_probe --case lu --size 12000 --matrix random --cores auto --timeout 120 --out runs/verification/native_blas/fixed_lu12000
python -m tools.blas_probe --case lu --size 12000 --matrix random --cores auto --threads 8 --timeout 120 --out runs/verification/native_blas/fixed_lu12000_threads8
python -m tools.blas_probe --case jax --size 4612 --cores auto --out runs/verification/native_blas/fixed_jax
python -m tools.web.solve_runner --surface runs/web/20261105-141850-a135fd/mesh/surface.json --out runs/verification/native_blas/fixed_web --wake fixed
```

已安装修复后的验收：

| 检查 | 实测 |
| --- | --- |
| 候选 DLL，1024 阶 DGEMM | auto / Haswell / Sandybridge 各 3/3 正确 |
| 候选 DLL，4612 阶随机 LU | 自动 Haswell，3/3 正确 |
| 8192 / 12000 阶随机 LU，单线程 | 各 3/3 正确 |
| 12000 阶随机 LU，8 线程 | 3/3 正确 |
| 4612 阶 JAX solve | 自动 Haswell，3/3 正确 |
| SciPy 单/双精度实数/复数 | 36/36 正确：K=127/128/129/255/256/257/513/1024 的 GEMM，加 513 阶随机制造解 |
| 原始 4598 壁面、4612 未知量机翼 | 完成；CL=0.5914475754705973，CD=0.007134760028134315，相对线性残差 2.65e-15 |
| 全量 pytest | 801 passed / 21 skipped / 11 failed，175.10 秒，无本次进程崩溃；11 项均在此前失败集合内 |

原机翼 runner 实测 65.71 秒，与大矩阵探针有部分同时运行，不作为速度比较。
全量测试结束后独立测得修复库的 4612 阶随机矩阵单线程纯 LU 三次中位数为
0.8215 秒（`fixed_lu4612`），与此前诊断 Q=128 的 0.8195 秒接近。
相对已有 Sandybridge 诊断对照，CL 差 7.77e-16，CD 差 4.16e-17，
面板 Cp 最大差 8.04e-14。原始故障证据全部保留。安装收据为
`runs/verification/native_blas/runtime_fix/receipt.json`，验收结果在上述各 `--out` 目录，
机翼比较为 `fixed_web_comparison.json`。这些是运行库正确性和代数验收，不是物理精度验收。
全量回归命令为 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q test -p no:cacheprovider`（不设置 `OPENBLAS_CORETYPE`），原始失败输出
保留在 `fixed_pytest.log`；未修复这些既有的梯度、递归和压力地板测试失败。

**Q=128 可以避免本次越界，即使总未知量继续增加。** GEMM 驱动把 K 切成不大于 Q 的
块；N 不需要小于 128，N 增大增加的是块数。它不能消除所有大规模限制：稠密矩阵内存
为 O(N²)，LU 计算量为 O(N³)，一张 12000 阶 double 矩阵约 1.07 GiB，
20000 阶约 2.98 GiB；实际装配/分解会同时持有多张数组。此次最大实测到 12000 阶随机
系统，实际机翼只重跑到 4612 阶，不宣称更大完整网格已验收。

热修复不会改变 conda 包版本记录，未来重装/更新 `libopenblas` 可能覆盖它。
届时优先使用含上述上游修正的正式构建，并重新运行探针；脚本对未知哈希拒绝改写。
需要恢复原始文件时，可从原 conda 包重装，但 build 0 的原始缺陷也会恢复。
