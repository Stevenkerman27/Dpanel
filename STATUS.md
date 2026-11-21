# 项目状态

更新：2026-11-21。本文件是根目录唯一的状态入口；历史交接见 `docs/archive/`。

## 一句话

低速主线是 Torch P1 源/偶极子面元。Web 默认固定尾迹、Galerkin 弱壁面、`pressure` Kutta 和 `area` 后缘速度恢复。固定尾迹、自由尾迹内层与固定拓扑的几何伴随共用 `area` 默认值。**物理精度尚未验收**（`physical_accuracy_accepted=false`）。

## 路线复核（2026-11-21）

- 当前主线与路线边界：固定尾迹 P1 有前向和几何伴随；自由尾迹有材料线推进与几何门禁，但缺跨构型平衡验收和耦合伴随。全势/P2 有受限算例前向证据，通用三维形状梯度与跨声速格式未完成。JAX、升力线和 VSPAERO 用作独立对照；WBCAero 仅作路线参考。
- 全量回归采用 Windows/myml、Python 3.13.14、Torch 2.13.0+cu132、NumPy 2.4.6、SciPy 1.18.0，BLAS/OMP/MKL 单线程。执行 `python -m pytest -q -p no:cacheprovider --tb=short test`，结果 **677 passed / 17 skipped / 2 xfailed / 2 failed，370.58 s**。两项失败均发生在读取历史 CAD 输入时，缺少 `runs/cases/wingbody/half_cad_wingbody_v1/surface.json` 和 `runs/cases/wingbody/half_cad_wingbody_3x/surface.json`。两项预期失败仍是 `pressure` 配点闭合的升力斜率和弦向细化。全量套件收集后新增的闭合 Jacobian 回归随 `test_p1_potential_jump_newton.py` 和 `test_p1_cst_gradient.py` 一并复跑，26 项通过。
- 历史后缘恢复对照使用同一 30° 后掠 GRASS 翼、AR 6、4°、782 个壁面节点和固定尾迹。下表为 `CL / Trefftz CDi / Σ|ΔΓ|`。同一后掠翼在弦向 6 分段、每段展向 3/4/5 等分下，`face + potential_jump` 的 e = 0.377 / 0.099 / 0.0028，`area + potential_jump` 的 e 稳定在 0.895–0.899。生产接口已删除 `face`，历史对照和冻结输入继续留档。见 [模型定义](docs/MODEL.md) §5–§6 和 [历史对照](docs/VALIDATION.md) §3。

| 壁面与恢复 | `pressure` | `potential_jump` |
| --- | --- | --- |
| collocation + face | 0.278962 / 0.004218 / 0.4545 | 0.345508 / 0.039837 / 6.9725 |
| Galerkin + face | 0.298891 / 0.004936 / 0.3672 | 0.343703 / 0.060171 / 8.7622 |
| Galerkin + area | 0.299383 / 0.004979 / 0.2506 | 0.299370 / 0.004980 / 0.2503 |

复算当前 `area` 行，在项目指定的 `myml` Python 环境和仓库根目录运行。旧 `face` 两行使用 `8807b61`。

```python
import numpy as np
import torch
from panel.p1_lifting_g1 import solve
from tools.wake_closure_study import build, _trefftz

torch.set_num_threads(1)
wing, digest = build("swept30")
print(len(wing.faces), digest)
for kutta in ["pressure", "potential_jump"]:
    state = solve(wing, 4., kutta=kutta)
    print(kutta, state["CL"], _trefftz(wing, state)["CDi_ideal"],
          np.abs(np.diff(state["gamma"])).sum())
```

## 当前状态

- **固定尾迹与优化：** GRASS/CST 固定拓扑支持隐式梯度；尚需在多构型、网格细化下验收压力、环量、载荷和方向导数。梯度优化的计算成本与规模仍受稠密装配限制。
- **共用内层与反传：** 固定和面片自由尾迹共用 `solve_multi`。壁面固定时，尾迹推进只重装 `W`。固定尾迹伴随删除重复的 NumPy 配点影响矩阵装配，保留源面块检查点，复用前向求解的伴随因子。Web 与 GRASS 优化将多载荷梯度合入一次批量 VJP。
- **反传基准：** Windows/myml，CPU 单线程，792 面 30° 后掠翼，迎角 4°，`area`，返回 `CL` 与精确 Trefftz `CDi` 的两行梯度，各取 3 次的中位数。`pressure` 从 38.398 s 降至 24.663 s，载荷最大绝对差为 0，梯度为 1.261e-7。`potential_jump` 从 41.582 s 降至 27.085 s，载荷最大绝对差为 2.539e-13，梯度为 5.677e-12。梯度比较涵盖壁面坐标、尾迹坐标和参考面积。脚本、前后收据位于 `runs/scratch/area_adjoint_bench.py`、`area_adjoint_before.json`、`area_adjoint_after.json`。Web GRASS 两项测试通过。Web 优化和四截面研究脚本各完成两次目标与梯度评估，按设定在一次 SLSQP 迭代后达到迭代上限。
- **尖翼梢对照：** 3096 面椭圆翼，4°，配点壁面，`potential_jump` 在 `area` 下测得闭合 Jacobian 条件数 16.1，历史 `face` 约 1.6e6。同一几何仍触发后缘后掠告警。奇异根拒绝由独立的代数测试覆盖。
- **自由尾迹：** 结果依配置而异，部分 CAD 案例会几何阻塞或载荷异常。收敛必须同时检查未松弛映射残差、法向滑移、尾迹压差和几何有效性，不能只看载荷或迭代状态。
- **物理校准：** 标模套件已有球体、薄翼和椭圆翼结果；二维圆柱、Joukowski 与 NACA 0012 的统一校准仍缺入口或证据。固定尾迹 P1 优先完成跨网格、跨平面的独立验证。
- **全势/P2：** 已有受限二维升力翼型、三维闭体和翼身前向算例；非线性跨声速方案未通过冻结验收，通用 CAD 体网格和几何伴随仍未完成。
- **GitHub：** 2026-11-21 复核时远端 master 为 `fdb02c8`，本地 `8807b61` 包含 3 个后续文档提交。开放 36 个 issue、2 个 PR（#114、#98）。#115 的标模校准与 #103 的 P1 几何可微边界仍有后续工作。

## 文档入口

| 想知道 | 看 |
| --- | --- |
| 协作、分支、验收规范 | [AGENTS.md](AGENTS.md) |
| 现役文档与证据索引 | [docs/README.md](docs/README.md) |
| 模型定义：方程、离散、两种闭合、后缘恢复、力与可微 | [docs/MODEL.md](docs/MODEL.md) |
| 数值定义：装配、约束降维、求解器、判据、条件数、成本 | [docs/NUMERICS.md](docs/NUMERICS.md) |
| 接口契约：调用链、默认值、输出字段、异常、数据位置 | [docs/INTERFACE.md](docs/INTERFACE.md) |
| 参照值：冻结包、A/B 表、细化、回归基线、不声明项 | [docs/VALIDATION.md](docs/VALIDATION.md) |
| 缺陷登记：#77 两条、#92 移植、face+pj 缺口、开放 issue | [docs/DEFECTS.md](docs/DEFECTS.md) |
| #77/#92 的原始数据与过程日志 | `verification/issue92/`、`verification/issue77/`（测试重放用）；日志在 [docs/archive/issue/](docs/archive/issue/) |
| 标模校准与验收比较 | [docs/reference/CALIBRATION.md](docs/reference/CALIBRATION.md)、[docs/reference/ACCEPTANCE.md](docs/reference/ACCEPTANCE.md) |
| 尾迹模型和自由尾迹限制 | [docs/reference/ARCHITECTURE_WAKE_MODELS.md](docs/reference/ARCHITECTURE_WAKE_MODELS.md) |
| 几何可微性 | [docs/reference/GEOMETRY_DIFFERENTIABILITY.md](docs/reference/GEOMETRY_DIFFERENTIABILITY.md) |
| 测试状态 | [docs/reference/TEST_STATUS.md](docs/reference/TEST_STATUS.md) |
| 退役代码及历史研究 | [docs/RETIRED_RESEARCH.md](docs/RETIRED_RESEARCH.md)、[docs/archive/README.md](docs/archive/README.md) |
