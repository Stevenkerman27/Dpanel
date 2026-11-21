# 项目状态

更新：2026-11-21。本文件是根目录唯一的状态入口；历史交接见 `docs/archive/`。

## 一句话

低速主线是 Torch P1 源/偶极子面元。Web 默认固定尾迹、Galerkin 弱壁面、`pressure` Kutta 和 `area` 后缘速度恢复。固定尾迹、自由尾迹内层与固定拓扑的几何伴随共用 `area` 默认值。**物理精度尚未验收**（`physical_accuracy_accepted=false`）。

## 路线复核（2026-11-21）

- 当前主线与路线边界：固定尾迹 P1 有前向和几何伴随；自由尾迹有材料线推进与几何门禁，但缺跨构型平衡验收和耦合伴随。全势/P2 有受限算例前向证据，通用三维形状梯度与跨声速格式未完成。JAX、升力线和 VSPAERO 用作独立对照；WBCAero 仅作路线参考。
- 现役 pytest 移除了 #92 端到端冻结回放、S1/S2/S3/S5 冻结答案、Torch 与旧 SciPy 求解器并排测试及旧载荷断言。测试保留解析极限、方程与约束、中心差分梯度和接口检查。原算例和移植过程见 [历史离散化对照](docs/archive/notes/DISCRETIZATION_COMPARISONS.md)。
- Windows/myml，Python 3.13.14、Torch 2.13.0+cu132、NumPy 2.4.6、SciPy 1.18.0，BLAS/OMP/MKL 单线程。`python -m pytest -q -p no:cacheprovider --strict-markers --tb=short -m "not macro" test` 为 **660 passed / 8 skipped**（196.08 s）。同参数的 `-m macro` 为 **8 passed / 2 xfailed**（139.98 s）。两项预期失败覆盖 `pressure` 配点闭合的升力斜率和弦向细化。

## 当前状态

- **固定尾迹与优化：** GRASS/CST 固定拓扑支持隐式梯度；尚需在多构型、网格细化下验收压力、环量、载荷和方向导数。梯度优化的计算成本与规模仍受稠密装配限制。
- **共用内层与反传：** 固定和面片自由尾迹共用 `solve_multi`。壁面固定时，尾迹推进只重装 `W`。固定尾迹伴随删除重复的 NumPy 配点影响矩阵装配，保留源面块检查点，复用前向求解的伴随因子。Web 与 GRASS 优化将多载荷梯度合入一次批量 VJP。
- **反传基准：** Windows/myml，CPU 单线程，792 面 30° 后掠翼，迎角 4°，`area`，返回 `CL` 与精确 Trefftz `CDi` 的两行梯度，各取 3 次的中位数。`pressure` 从 38.398 s 降至 24.663 s，载荷最大绝对差为 0，梯度为 1.261e-7。`potential_jump` 从 41.582 s 降至 27.085 s，载荷最大绝对差为 2.539e-13，梯度为 5.677e-12。梯度比较涵盖壁面坐标、尾迹坐标和参考面积。脚本、前后收据位于 `runs/scratch/area_adjoint_bench.py`、`area_adjoint_before.json`、`area_adjoint_after.json`。Web GRASS 两项测试通过。Web 优化和四截面研究脚本各完成两次目标与梯度评估，按设定在一次 SLSQP 迭代后达到迭代上限。
- **尖翼梢：** 3096 面椭圆翼，4°，配点壁面，`potential_jump + area` 的闭合 Jacobian 条件数为 16.1。该几何仍触发后缘后掠告警。奇异根拒绝由代数测试覆盖。
- **自由尾迹：** 结果依配置而异，部分 CAD 案例会几何阻塞或载荷异常。收敛必须同时检查未松弛映射残差、法向滑移、尾迹压差和几何有效性，不能只看载荷或迭代状态。
- **物理校准：** 球体最细网格解析 \(C_p\) RMS 误差为 0.01316。默认固定尾迹薄翼升力斜率较升力线理论低 8.46%。椭圆翼 AR 6 三档网格由 272 面增至 1560 面，展向效率从 0.9737 升至 0.9819，升力偏差为 −1.81%、−1.80%、−2.32%。二维圆柱和 Joukowski 有独立原型的解析对照。复现和详细结果见 [标模校准](docs/reference/CALIBRATION.md)。
- **测试分层：** 新增默认主线的球体、薄翼、椭圆翼和零迎角自由尾迹四项宏观检查。零迎角自由尾迹通过收敛判据，2° 工况四步后映射残差为 0.0548，仍待验证有载尾迹平衡。NACA 0012 风洞数据仅作归档。[测试与验证](docs/VALIDATION.md)
- **VSPAERO：** OpenVSP 3.56.0 完成四构型、两档网格、每档四个迎角。细网格下矩形翼和椭圆翼的升力斜率差为 −5.41% 和 +0.15%。后掠和翼身融合体用于诊断，VSPAERO 重建省略弯度。翼身融合体的环量 RMS 差为 0.409，仍待对齐几何和条带口径。报告在 `runs/verification/current_vspaero/summary.json`。
- **全势/P2：** 已有受限二维升力翼型、三维闭体和翼身前向算例；非线性跨声速方案未通过冻结验收，通用 CAD 体网格和几何伴随仍未完成。
- **GitHub：** 2026-11-21 核对为 36 个开放 issue、2 个开放 PR。#115 的标模校准和 #103 的 P1 几何可微边界仍开放。上一轮推送被 GitHub 拒绝，现有令牌缺少提交链中 GitHub Actions 改动所需的 `workflow` 权限。

## 文档入口

| 想知道 | 看 |
| --- | --- |
| 协作、分支、验收规范 | [AGENTS.md](AGENTS.md) |
| 现役文档与证据索引 | [docs/README.md](docs/README.md) |
| 模型定义：方程、离散、两种闭合、后缘恢复、力与可微 | [docs/MODEL.md](docs/MODEL.md) |
| 数值定义：装配、约束降维、求解器、判据、条件数、成本 | [docs/NUMERICS.md](docs/NUMERICS.md) |
| 接口契约：调用链、默认值、输出字段、异常、数据位置 | [docs/INTERFACE.md](docs/INTERFACE.md) |
| 实验、理论、数值及外部求解器验证入口 | [docs/VALIDATION.md](docs/VALIDATION.md) |
| 缺陷登记 | [docs/DEFECTS.md](docs/DEFECTS.md) |
| #77/#92 的原始数据与过程日志 | `verification/issue92/`、`verification/issue77/`，日志在 [docs/archive/issue/](docs/archive/issue/) |
| 标模校准与验收比较 | [docs/reference/CALIBRATION.md](docs/reference/CALIBRATION.md)、[docs/reference/ACCEPTANCE.md](docs/reference/ACCEPTANCE.md) |
| 尾迹模型和自由尾迹限制 | [docs/reference/ARCHITECTURE_WAKE_MODELS.md](docs/reference/ARCHITECTURE_WAKE_MODELS.md) |
| 几何可微性 | [docs/reference/GEOMETRY_DIFFERENTIABILITY.md](docs/reference/GEOMETRY_DIFFERENTIABILITY.md) |
| 2026-11-03 测试记录 | [docs/reference/TEST_STATUS.md](docs/reference/TEST_STATUS.md) |
| 退役代码及历史研究 | [docs/RETIRED_RESEARCH.md](docs/RETIRED_RESEARCH.md)、[docs/archive/README.md](docs/archive/README.md) |
