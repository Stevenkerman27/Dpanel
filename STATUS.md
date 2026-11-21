# 项目状态

更新：2026-11-21。本文件是根目录唯一的状态入口；历史交接见 `docs/archive/`。

## 一句话

低速主线是 Torch P1 源/偶极子面元。Web 默认固定尾迹、Galerkin 弱壁面、`pressure` Kutta 和 `face` 恢复；GRASS/CST 固定拓扑已接通隐式几何梯度。**物理精度尚未验收**（`physical_accuracy_accepted=false`）。

## 路线复核（2026-11-21）

- 当前主线与路线边界：固定尾迹 P1 有前向和几何伴随；自由尾迹有材料线推进与几何门禁，但缺跨构型平衡验收和耦合伴随。全势/P2 有受限算例前向证据，通用三维形状梯度与跨声速格式未完成。JAX、升力线和 VSPAERO 用作独立对照；WBCAero 仅作路线参考。
- 全量回归：Windows/myml、Python 3.13.14、Torch 2.13.0+cu132、NumPy 2.4.6、SciPy 1.18.0；单线程 BLAS/OMP/MKL 执行 `python -m pytest -q test -p no:cacheprovider`，结果 **674 passed / 17 skipped / 2 xfailed / 2 failed，299.39 s**。两项失败是冻结回归所需的翼身 CAD 输入已不存在；求解未开始。两项预期失败是 `pressure` 配点闭合的升力斜率和弦向细化问题。
- 后缘恢复对照：同一 30° 后掠 GRASS 翼、AR 6、4°、782 个壁面节点、固定尾迹。下表为 `CL / Trefftz CDi / Σ|ΔΓ|`。此单网格结果显示 `area` 恢复显著减弱环量锯齿、使两种闭合接近。2026-11-21 补充展向细化：同一后掠翼（弦向 6 分段）每段 3/4/5 等分下，`face + potential_jump` 的 e = 0.377 / 0.099 / 0.0028（CDi 到 1.6e1，Γ 变号 56 次），`area + potential_jump` 稳定在 0.895–0.899；即坏组合是 “face + pj”，不是 pj 方程或后掠翼本身。结论、对照表与复算入口见 [docs/TRAILING_EDGE.md](docs/TRAILING_EDGE.md)；尚不能推广为精度结论。

| 壁面与恢复 | `pressure` | `potential_jump` |
| --- | --- | --- |
| collocation + face | 0.278962 / 0.004218 / 0.4545 | 0.345508 / 0.039837 / 6.9725 |
| Galerkin + face | 0.298891 / 0.004936 / 0.3672 | 0.343703 / 0.060171 / 8.7622 |
| Galerkin + area | 0.299383 / 0.004979 / 0.2506 | 0.299370 / 0.004980 / 0.2503 |

复算：在项目指定的 `myml` Python 环境、仓库根目录运行：

```python
import numpy as np
import torch
from panel.p1_lifting_g1 import solve
from tools.wake_closure_study import build, _trefftz

torch.set_num_threads(1)
wing, digest = build("swept30")
print(len(wing.faces), digest)
for wall, recovery in [("collocation", "face"), ("galerkin", "face"), ("galerkin", "area")]:
    for kutta in ["pressure", "potential_jump"]:
        state = solve(wing, 4., wall_formulation=wall, pressure_recovery=recovery, kutta=kutta)
        print(wall, recovery, kutta, state["CL"], _trefftz(wing, state)["CDi_ideal"],
              np.abs(np.diff(state["gamma"])).sum())
```

## 当前状态

- **固定尾迹与优化：** GRASS/CST 固定拓扑支持隐式梯度；尚需在多构型、网格细化下验收压力、环量、载荷和方向导数。梯度优化的计算成本与规模仍受稠密装配限制。
- **自由尾迹：** 结果依配置而异，部分 CAD 案例会几何阻塞或载荷异常。收敛必须同时检查未松弛映射残差、法向滑移、尾迹压差和几何有效性，不能只看载荷或迭代状态。
- **物理校准：** 标模套件已有球体、薄翼和椭圆翼结果；二维圆柱、Joukowski 与 NACA 0012 的统一校准仍缺入口或证据。固定尾迹 P1 优先完成跨网格、跨平面的独立验证。
- **全势/P2：** 已有受限二维升力翼型、三维闭体和翼身前向算例；非线性跨声速方案未通过冻结验收，通用 CAD 体网格和几何伴随仍未完成。
- **GitHub：** 2026-11-21 复核时远端 master 为 `7a959d9`，本地提交比远端领先 6 个；开放 36 个 issue、2 个 PR（#114、#98）。#115 的标模校准与 #103 的 P1 几何可微边界仍有后续工作。

## 文档入口

| 想知道 | 看 |
| --- | --- |
| 协作、分支、验收规范 | [AGENTS.md](AGENTS.md) |
| 现役文档与证据索引 | [docs/README.md](docs/README.md) |
| 实际求解路线、各入口默认值、已知不一致 | [docs/ROUTE.md](docs/ROUTE.md) |
| 后缘闭包（pressure/pj）与速度恢复（face/area）怎么选 | [docs/TRAILING_EDGE.md](docs/TRAILING_EDGE.md) |
| issue #77（接触迹/Γ 支撑）与 #92（torch 移植）留下了什么 | [docs/ISSUE77_92.md](docs/ISSUE77_92.md) |
| 标模校准与验收比较 | [docs/CALIBRATION.md](docs/CALIBRATION.md)、[docs/ACCEPTANCE.md](docs/ACCEPTANCE.md) |
| 尾迹模型和自由尾迹限制 | [docs/ARCHITECTURE_WAKE_MODELS.md](docs/ARCHITECTURE_WAKE_MODELS.md) |
| 几何可微性 | [docs/GEOMETRY_DIFFERENTIABILITY.md](docs/GEOMETRY_DIFFERENTIABILITY.md) |
| 测试状态 | [docs/TEST_STATUS.md](docs/TEST_STATUS.md) |
| 退役代码及历史研究 | [docs/RETIRED_RESEARCH.md](docs/RETIRED_RESEARCH.md)、[docs/archive/README.md](docs/archive/README.md) |
