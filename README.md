# 快速势流求解器研究实现

截至 2026-10-26，本地执行与云端协同都在进行中（GitHub 监听常驻，云端按 issue/PR 下验收契约）。**已完成机身跨后缘、粗机身与双机身的可微面元实现及公开矩阵**：37项主检查、10项补充拓扑检查和180项AD/中心差分通过；在此基础上完成了顶点/站位几何 VJP、Torch↔JAX 桥、核函数分块内存改造、矩阵无关 GMRES 与近场预条件、以及 OCC 正半域镜像半模。方法、原始误差与复现命令见 [docs/README.md](docs/README.md)（文档索引），最新鲁棒性证据见 [PANEL_ROBUSTNESS.md](docs/PANEL_ROBUSTNESS.md)，`panel/` 入口见 [panel/README.md](panel/README.md)。

新 `panel/` 求解器采用解析源/偶极子、真实流体侧尾迹、线性势跳跃及接触相容基；纯表面CAD流程避开体网格优化。没有压力裁剪或矩阵正则化。完整通用求解器尚未验收，独立壁面误差与粗细载荷差仍显著，`physical_accuracy_accepted=false`。此前仅支持前置机身的VortexAD闭环保留于 [WINGBODY_PANEL_LOOP.md](docs/WINGBODY_PANEL_LOOP.md)；总体状态见 [DEVELOPMENT.md](DEVELOPMENT.md)，路线选择见 [SOLVER_ROUTE.md](SOLVER_ROUTE.md)。

当前开发优先级为几何鲁棒性：厚翼身连接、机身越过后缘、接近相切时的网格拓扑和明确失败诊断。暂缓昂贵的精度加密；原始压力、壁面误差和所有旧失败记录继续保留。

## 目录结构

| 路径 | 内容 | 入口文档 |
| --- | --- | --- |
| `panel/` | JAX 可微面元求解器（CAD→网格→源/偶极子→尾迹/Kutta→载荷→AD/VJP→矩阵无关→半模） | [panel/README.md](panel/README.md) |
| `torch_panel/` | 从 GRASS 提取的可微 Torch 面元内核 | [torch_panel/README.md](torch_panel/README.md) |
| `core/` `potential/` `volume/` `airfoil2d/` `lifting3d/` | 旧一代求解器与研究方法（保留对照与溯源） | 各目录 `README.md` |
| `test/` | 后端探针与迁移测试（VortexAD、线性求解器、torch 迁移、翼身回路） | [test/README.md](test/README.md) |
| `cases/` | 公开表面/CAD 夹具（`dpanel.planar_cut.v1` 等） | [panel/README.md](panel/README.md) |
| `monitor/` | GitHub 监听（commit/PR/issue/discussions）与发评论脚本 | [monitor/README.md](monitor/README.md) |
| `docs/` | 跨模块设计文档、云端验收报告、证据快照 | [docs/README.md](docs/README.md) |
| `runs/` | 可复现运行产物（`_logs/` 控制台日志、`_bench_kernel/` 分块基准、各验收算例、git worktree） | 各 `runs/<case>/report.json` |
| `archive/` | 被取代的文档副本、一次性调试脚本、安装日志（不入库） | [archive/README.md](archive/README.md) |
| `audit/` `upgrade/` `OpenVSP-3.52.1-win64/` `WBCAERO/` `.tools/` | 外部工具链与上游调研快照（不入库） | [audit/REPORT.md](audit/REPORT.md) |

`runs/` 只放运行产物，不放入库文档；一次性脚本已归档到 `archive/scripts/oneoff/`，仍被文档引用的 27 个脚本留在 `runs/` 根目录。

## 单算例入口

在已有 NumPy、SciPy、threadpoolctl 的 Python 环境中：

```powershell
python aero.py --case cases/panel_two_bodies.json --out runs/my_panel --export-surface
python aero.py --case cases/panel_cad_wingbody.json --out runs/my_panel_cad --export-surface
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

`panel-cut`入口需要JAX环境（本机 `.conda_env/python.exe`），使用公开表面JSON夹具时不需要CAD依赖；其余入口沿用原NumPy/SciPy环境。输出目录必须为空。入口保存输入副本、mesh/solve 原始命令、完整日志、输入/代码哈希、阶段耗时和 `manifest.json`。网格与解保存在 `mesh/` 和 `solution/`；已有网格则直接引用原路径。数值收敛或几何门禁通过均不能解释为任意算例已通过物理验收。

当前机器的 Python 为 `C:/Users/zyx20/anaconda3/python.exe`。Gmsh 4.15.2 API/DLL 位于 `audit/runtime/python/`，网格脚本显式加载这个本地依赖；原 OpenVSP/WBCAero 二进制保持原样。默认请求 NCPU=8；SciPy SuperLU 的稀疏直接分解不能因此宣称使用8线程。

`--export-surface`导出三维升力算例的原始Tri6表面、面均Cp、逐面压力积分及独立原始高斯点VTU；机身/机翼标签保留。面均值只用于显示，压力积分直接使用原始高斯点；见 `lifting3d/EXPORT.md`。manifest同时保存原始壁面/尾迹诊断和组件力，三维升力的 `physical_accuracy_accepted` 当前仍为false。

三维升力入口同时输出 `pressure_loads.json`，给出机翼/机身/总压力力和三轴力矩。默认力矩中心为翼根四分之一弦，可用 `numerics.moment-center` 指定保存网格坐标中的中心；分母为实际Sref乘根弦。解析压力场、尺度与参考点变换检查见 `lifting3d/LOADS.md`。

三维升力入口在求解前运行独立网格检查，manifest分别记录 `geometry_accepted` 和数值收敛。可用 `numerics.linear-solver="schur"` 选择精确分块；该后端不正则化或截断环量未知量，同源码两网格对照14/14通过，但未显示速度优势。

## 实现与证据

| 路径 | 方法与当前范围 | 验证记录 |
|---|---|---|
| `panel/solve.py`、`panel/cad.py` | 可微低速源/偶极子、纯表面CAD、跨后缘接触尾迹、分部件载荷 | [方法与证据](docs/PANEL_ROBUSTNESS.md)、[使用](panel/README.md)；物理精度未验收 |
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
