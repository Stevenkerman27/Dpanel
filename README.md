# 快速势流求解器研究实现

截至 2026-10-24，开发目标已按用户要求暂停，尚未完成整体验收。已完成工作、失败记录与恢复起点见 [PAUSE_HANDOFF.md](PAUSE_HANDOFF.md)。当前可复现二维翼型、三维闭体及固定平面尾迹的参数化/CAD 升力原型；三维局部物理精度、复杂尾迹及跨声速尚未验收。总体状态见 [DEVELOPMENT.md](DEVELOPMENT.md)，路线选择见 [SOLVER_ROUTE.md](SOLVER_ROUTE.md)。

当前开发优先级为几何鲁棒性：厚翼身连接、机身越过后缘、接近相切时的网格拓扑和明确失败诊断。暂缓昂贵的精度加密；原始压力、壁面误差和所有旧失败记录继续保留。

## 单算例入口

在已有 NumPy、SciPy、threadpoolctl 的 Python 环境中：

```powershell
python aero.py --case cases/naca0012.json --out runs/my_naca
python aero.py --case cases/wing_initial.json --out runs/my_wing
python aero.py --case cases/wing_strips.json --out runs/my_compressible_wing
python aero.py --case cases/wingbody_strips.json --out runs/my_wingbody
python aero.py --case cases/wingbody_aft.json --out runs/my_aft_wingbody --export-surface
python aero.py --case cases/wingbody_swept_aft.json --out runs/my_swept_aft_wingbody --export-surface
python aero.py --case cases/wing_swept.json --out runs/my_swept_wing --export-surface
python aero.py --case cases/cad_sphere.json --out runs/my_cad_body
python aero.py --case cases/cad_wingbody.json --out runs/my_lifting_cad --export-surface
```

输出目录必须为空。入口保存输入副本、mesh/solve 原始命令、完整日志、输入/代码哈希、阶段耗时和 `manifest.json`。网格与解保存在 `mesh/` 和 `solution/`；已有网格则直接引用原路径。`numerically_converged` 只表示离散方程收敛，`validation` 字段另行说明精度证据，不能将它解释为任意算例已通过物理验收。

当前机器的 Python 为 `C:/Users/zyx20/anaconda3/python.exe`。Gmsh 4.15.2 API/DLL 位于 `audit/runtime/python/`，网格脚本显式加载这个本地依赖；原 OpenVSP/WBCAero 二进制保持原样。默认请求 NCPU=8；SciPy SuperLU 的稀疏直接分解不能因此宣称使用8线程。

`--export-surface`导出三维升力算例的原始Tri6表面、面均Cp、逐面压力积分及独立原始高斯点VTU；机身/机翼标签保留。面均值只用于显示，压力积分直接使用原始高斯点；见 `lifting3d/EXPORT.md`。manifest同时保存原始壁面/尾迹诊断和组件力，三维升力的 `physical_accuracy_accepted` 当前仍为false。

三维升力入口同时输出 `pressure_loads.json`，给出机翼/机身/总压力力和三轴力矩。默认力矩中心为翼根四分之一弦，可用 `numerics.moment-center` 指定保存网格坐标中的中心；分母为实际Sref乘根弦。解析压力场、尺度与参考点变换检查见 `lifting3d/LOADS.md`。

三维升力入口在求解前运行独立网格检查，manifest分别记录 `geometry_accepted` 和数值收敛。可用 `numerics.linear-solver="schur"` 选择精确分块；该后端不正则化或截断环量未知量，同源码两网格对照14/14通过，但未显示速度优势。

## 实现与证据

| 路径 | 方法与当前范围 | 验证记录 |
|---|---|---|
| `potential/solve_airfoil_log.py` | 共享面通量的对数径向 FV，解析映射翼型，严格亚声速 | [LOG_RESULTS.md](potential/LOG_RESULTS.md) |
| `airfoil2d/solve.py` | 一般二维曲面P2、环量基、Kutta，严格亚声速 | [RESULTS.md](airfoil2d/RESULTS.md)、[追加失败报告](airfoil2d/EXTRA_RESULTS.md) |
| `volume/solve_p2.py` | 三维闭体曲面P2，无升力尾迹 | `volume/results/`，[P2_PROTOCOL.md](volume/P2_PROTOCOL.md) |
| `volume/build_cad_mesh.py` | STEP/BREP/IGES封闭实体、显式原生坐标映射、无升力闭体网格 | [CAD_RESULTS.md](volume/CAD_RESULTS.md)、[CAD_PROTOCOL.md](volume/CAD_PROTOCOL.md) |
| `lifting3d/solve.py` | 三维尾迹切割、相容条带势跳跃、节点/积分Kutta、机翼/机身压力载荷 | [结果](lifting3d/RESULTS.md)、[协议](lifting3d/PROTOCOL.md)，仍在精度验证 |
| `lifting3d/build_cad_mesh.py` | 完整闭合升力 CAD、弯度/不对称实体、显式平面后缘与局部拓扑切割 | [CAD 结果](lifting3d/CAD_ROBUSTNESS_RESULTS.md)、[CAD 契约](lifting3d/CAD_PROTOCOL.md)；9 个配置全流程通过，双机身仅修复后建网格成功 |
| `core/solve.py` | 独立低速面元参考 | 封闭体参考保留；早期三维升力表示未验收 |
| `potential/entropy_riemann.py` | E-O标量辅助问题的激波/稀疏扇与熵验证 | `potential/results/entropy_v1/`，不代表二维跨声速 |

P2 的节点弱式质量平衡、原始壁面梯度误差、内部面通量跳跃分别输出。不会裁剪压力、删除法向速度或将弱残差冒充点态无穿透。失败结果保留在原目录，修订后使用新目录和源快照。

## 模型边界

无黏、附着势流；压力阻力含离散误差和三维诱导贡献，不含摩擦阻力。分离、失速、强激波不能由当前模型可靠预测。跨声速候选通量须通过守恒与熵验收后才能开放；密度偏置候选已检出熵反例并拒绝，见 `potential/results/transonic_flux_v1/`。

参数化三维生成器支持对称矩形/梯形/后掠翼及可选椭球机身，无弯度或扭转。其跨后缘机身分支仍要求最大截面不晚于所有后缘站位、相容条带、平面尾迹和 weak Kutta；24 个不同几何配置通过，见 `lifting3d/GEOMETRY_STATUS.md`。

新增 `lifting-cad-p2` 方法导入完整闭合 CAD，不镜像上半几何；已验证弯度翼、翼身、缩放和平移、后掠渐缩，以及最大截面晚于后缘的一个机身配置。该进展不改变旧参数化分支的限制。CAD 后缘须位于归一化 z=0、严格按 y 递增且 x(y) 分段线性，只用 weak Kutta。双机身默认粗网格曾在优化阶段失败，曲率约束重试只完成建网格，尚无全流程结论。精确相切、不可分辨接触、非平面后缘和复杂自由尾迹仍未普遍支持。最新结果与明确失败见 `lifting3d/CAD_ROBUSTNESS_RESULTS.md`。

三维新算例显式选择 `wake-strips`，使势跳跃基与尾迹面相容；0保留旧非结构化插值以复现历史结果。可选 `numerics.kutta="weak"` 是帽函数积分式后缘约束，目前与默认node并列作离散对照。图式尾迹迭代为独立研究驱动，两种内部网格运动的首轮均未通过，不纳入默认单算例入口。

CAD例子读取 `volume/test_cad_workflow.py` 已生成的STEP球体。自己的闭体输入修改case中的 `geometry.cad`；网格和输出使用CAD包围盒最大边长归一化的坐标，manifest保留原生坐标/面积变换。此路径仍是无升力闭体，不能据此自动求解机翼升力。

CAD原生原点是OCC保守包围盒中心，B-spline控制网的包围盒可大于真实几何范围；它不是质量中心。所有坐标按记录的可逆变换解释。周期缝线输入导致空物面时明确拒绝，保留空面编号及网格日志。原生椭球验证和失败输入见 `volume/ELLIPSOID_RESULTS.md`。

自适应背景场采用NPZ的 `points/tets/vertex_size`，可经wing case的 `geometry.background-view` 相对case路径传入；与原几何尺寸场取Min。该PostView分支为规避已复现的Gmsh多线程停滞，网格生成使用1线程并记录，流场仍请求8线程。见 `lifting3d/LIFTING_ADAPTATION_PROTOCOL.md`。

厚翼身全壁面0.5h加密已因内存预算终止，记录为 `resource_rejected`；只生成并检查了网格，没有加密后流场。当前不继续此精度路线。

## P1 continuous research solver (integration branch)

This branch introduces the continuous P1 doublet solver that previously existed only on
research branches. It is a research path, not a production solver: `physical_accuracy_accepted=false`.

Requirements: Python 3.11 with `numpy` and `jax` (CPU is enough). Run from the repository root.

    # 1) analytic sphere (P1 G0, closed body)
    python -m panel.run_p1_closed --levels 1 2 3 --samples 96 --out /tmp/g0

    # 2) generated single wing with a sharp trailing edge (P1 G1)
    python -m panel.run_p1_lifting_g1 --chord-panels 4 --span-panels 6 --alpha 5 \
        --kutta signed_speed --out /tmp/g1

    # 3) real half-model CAD (mirrored to a full wing, TE pairing from upper/lower TE faces)
    python -m panel.run_p1_lifting_g1 --half-surface cases/half_cad_wingbody_v1/surface.json \
        --alpha 5 --kutta pressure --out /tmp/g1cad

    # 4) main wing + raised tail (P1 G2, fixed wake)
    python -m panel.run_p1_lifting_g2 --main-half-surface cases/half_cad_wingbody_v1/surface.json \
        --tail-span 1.8 --tail-chord .55 --tail-x 1.6 --tail-z .35 --alpha 5 \
        --kutta pressure --reference-area 4.0 --out /tmp/g2

    # 5) opt-in experimental free wake (NOT converged, see docs/P1_FREE_WAKE_PROBE_FIX.md)
    python -m panel.run_p1_steady_wake --main-half-surface cases/.../surface.json --alpha 5 \
        --iterations 4 --relaxation .05 --max-step-chord .01 --probe-floor-fraction .6 --out /tmp/fw

Conventions: input `surface.json` is a positive-y DPanel half model mirrored in y; coordinates
are in chord units of the case; `reference_area` comes from the surface file (watch the
declared-vs-planform mismatch documented in `docs/INTEGRATION_P1.md` §5.5); angles are in
degrees; velocities are dimensionless multiples of the freestream. Every CLI writes
`report.json` (raw Cp, pressure and circulation loads, wall/Kutta/jump residuals) into the
`--out` directory, which must not exist yet.

Tests: `pytest test/test_p1_*.py -q -p dsh_tmp_plugin` (62 passed). Full audit, acceptance
commands, real exit codes and the list of unresolved assumptions: `docs/INTEGRATION_P1.md`.
