# 路线：现在实际在跑的那条链

更新 2026-11-21。状态数字只认 [STATUS.md](../STATUS.md)；本文回答“代码实际走哪条路、
每一步在哪、默认值是什么、哪里还不一致”。

一句话：**低速气动主线 = Torch P1 偶极面元 + 固定尾迹 + Galerkin 弱壁面 + pressure
后缘闭包 + face 速度恢复；几何导数走隐式微分。** 其余（自由尾迹、势跳变闭包、area
恢复、全势/P2、升力线、VSPAERO）都是可选路径或对照工具，不在默认链上。

## 1. 一条链的七步

| # | 做什么 | 实现位置 | 现状 |
|---|---|---|---|
| 1 | 几何 → 网格 | GRASS 截面：`panel/grass_mesh.py`（`build_grass_mesh`、`load_grass_wing`）；CAD 半模：`panel/cad_half.py`（尾迹/机身接触分类在 `panel/p1_geometry_contact.py`）；机身按非升力湿壁接入 | 固定拓扑可复现；CAD 输入部分冻结样例已不在库里 |
| 2 | 离散 | 壁面 P1 偶极子 μ（每顶点 1 个 DOF）+ 每个尾迹的展向 Γ（P1 hat 基）；单翼 `solve` 先 `combine_wings` 成多组 | 生产 |
| 3 | 壁面方程 | Galerkin 弱形式 `assemble_galerkin_multi(mesh, order=3)`；配点版 `assemble_multi` 保留用于对照 | Galerkin 是默认 |
| 4 | 尾迹 | 固定尾迹（求解期间几何不变）；自由尾迹 `panel/p1_steady_free_wake.py::solve_steady_free_wake`（材料线推进 + 几何门禁） | 固定是默认；自由尾迹无耦合伴随 |
| 5 | 后缘条件 | `J` 行 `μ_up − μ_down = H Γ`（势跳变）；闭包 `pressure`（默认）或 `potential_jump` | 见 [TRAILING_EDGE.md](TRAILING_EDGE.md) |
| 6 | 速度与压力 | 速度恢复 `face`（默认）/ `area` / `quadratic` → Bernoulli → Cp | 见 [TRAILING_EDGE.md](TRAILING_EDGE.md) |
| 7 | 力与诊断 | CL、CL_gamma、CD_pressure、CM_y；Trefftz 精确片线积分给 CDi/e；`physical_accuracy_accepted=False` | 门禁有，精度未验收 |

求解核只有一个：`panel/p1_lifting_g2.py::solve_multi`（默认 `kutta=None → PRODUCTION_KUTTA`、
`wall_formulation=None → galerkin`、`pressure_recovery="face"`），其后调
`panel/p1_solve_torch.py::solve_multi_torch`。`panel/p1_lifting_g1.py::solve` 只是
`combine_wings([wing])` 的薄适配器，单翼 torch 核 `solve_single_torch` 已删除。

## 2. 默认值按入口（这是最容易踩的坑）

| 入口 | 壁面 | 闭包 | 恢复 | 说明 |
|---|---|---|---|---|
| `p1_lifting_g2.solve_multi` | galerkin | pressure | face | 真正的默认源 |
| `p1_lifting_g1.solve` | galerkin | pressure | face | 薄适配器 |
| `tools/web/solve_runner.solve_fixed` | galerkin | production→pressure | face | Web 前向 |
| `tools/web/grass_runner` | 同左 | production→pressure | face | Web 梯度 |
| `tools/web/optimize_runner` | — | pressure | face | UI 不提供 recovery 选择 |
| `test/test_p1_cst_gradient.py`、`test/test_lifting_line_studies.py` | galerkin | 显式传 | **area** | 研究口径 |

浏览器工作台（`tools/web/params.py` 的 `SCHEMA`）没有 closure / recovery 字段，
所以**用户点出来的每一个解都是 pressure + face**；想换只能改代码或走 CLI 脚本。

## 3. 可微链（Web 优化用的就是这条）

`panel/p1_cst_gradient.py`：

- `fixed_wake_geometry(wing, xyz, wake, reference_area, alpha_deg, max_nfev, kutta=None, pressure_recovery="face")`（:549）
- 前向：`_FixedWakeImplicit.forward`（:324）/ `_PotentialJumpImplicit.forward`（:456）做一次求解，
  并把平稳性残差记进 meta；
- 反向：`backward`（:394 / :502）用隐式函数定理，只对**当前几何的算子**做 VJP，不保存求解轨迹。
- 适用范围：固定拓扑的设计变量（CST 系数、GRASS 截面参数）。顶点级 VJP 没有做，
  `docs/GEOMETRY_DIFFERENTIABILITY.md` 里的 JAX 实验是另一条线，未进生产。

## 4. 自由尾迹

`solve_steady_free_wake`（`panel/p1_steady_free_wake.py:537`）推进材料线、重建尾迹、重解，
带 `wake_material_graph_audit`、长宽比/交叉门禁和快照。现状：能跑、有几何门禁、**没有**
耦合伴随；跨构型（BWB、大迎角、真离面）的验收没做。#113 报的“自由尾迹 CL 崩 77%”在
现有网格上复现不出来（配置不同），需要在 `potential_jump` 默认下按 E1–E4 协议重测。

## 5. 已知不一致：文档说一套、代码跑一套

1. **恢复默认值**。`panel/p1_te_recovery.py` 的 docstring 写 “Area-weighted one-ring recovery
   is the production option”，但三个入口的默认都是 `face`。按 STATUS 与本文 §6 的复核，
   `area` 在同等成本下把后掠翼的环量锯齿压掉一个量级，且两闭包趋于一致。
   要么把默认改 `area`（要重跑冻结样例与 `test_p1_cst_gradient` 的一致性断言），
   要么把 docstring 改成事实。
2. **闭包默认值的 docstring 过期**。`p1_cst_gradient.py` 的 `fixed_wake_cst`(:195)、
   `fixed_wake_sections`(:210)、`fixed_wake_geometry`(:549) 都写着 “kutta=None is the
   production closure, potential_jump”，而 `PRODUCTION_KUTTA='pressure'`。这是 2026-11-10
   短暂翻过默认又改回时漏改的。
3. **研究口径与生产口径分叉**。测试用 `area`，生产用 `face`，于是“测试通过”不代表
   Web 上那条路径被同一组数字覆盖。
4. **旧索引把日志当入口**。`docs/KUTTA_CONDITION.md` 是 814 行的按时间追加日志，
   被当成后缘条件的唯一入口——现在它降级为证据日志，结论进 [TRAILING_EDGE.md](TRAILING_EDGE.md)。

## 6. 复现

```bash
# 单次生产解（Web 同款默认）
python -m panel.p1_lifting_g1 --help    # 或直接用 python -c ...
# 三条平面形状 × 闭包 × 尾迹的对照表
python -m tools.wake_closure_study --iterations 10
# 后掠/椭圆的恢复 × 闭包 A/B（本文数字）
python runs/verification/route_check/run.py
```

解释器固定 `C:/Users/zyx20/anaconda3/envs/myml/python.exe`，单线程 BLAS。
同一进程里 torch 之后再调 SciPy LAPACK 会 0xC0000005（见
`docs/archive/issue/ISSUE92_PHASE1_WAKE_EVALUATOR.md` §6f），证据脚本要分开进程跑。

## 7. 不在生产链上的东西

`potential/`（全势/P2）、`lifting3d/`（升力线 + CAD）、`core/`（JAX 面元）、`volume/`：
对照或研究代码，只有受限前向证据。WBCAero 只作路线参考，OpenVSP/VSPAERO 只作外部对照
（`vspapi/`，见 `docs/VSPAPI.md`）。不要把它们的默认当成主线的默认。
