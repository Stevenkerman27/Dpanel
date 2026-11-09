# 后缘 Kutta 条件：现状、理论、实测与两级求解方案

2026-11-09 四步修复与实际验证见 §10，修复前复核见 §9；两节纠正前文的效率定义、权重、
「势跳变闭包不使用压差」和「不同结果必然是同一方程多根」等解释。前文数字保留为历史证据。

状态：比较定义、固定环量探针、可选尾缘恢复与 Newton 诊断已实现；物理精度未验收。
适用代码：`panel/p1_cst_gradient.py`（`fixed_wake_geometry`）、`panel/p1_pressure_kutta_torch_driver.py`、`panel/p1_lifting_g1.py`（`assemble`）。
证据脚本：`runs/scratch_opt/morino_probe.py`、`runs/scratch_opt/itkut_probe.py`、`runs/scratch_opt/kutta_diag.py`。

## 1. 问题

当前升力求解把 Γ（尾迹偶极强度）当成未知量，用 `solve_pressure_kutta_torch` 让后缘上下表面的
动压差 `p = 0.5(|V_top|² − |V_bot|²)` 归零，同时最小化壁面方程残差。Γ 的个数只有 9~81，但
求的是**整个状态**（77~3403 个自由度）上的非线性最小二乘，因此

* 隐式求导要对 903×903（0.1 m）甚至 3403×3403（0.05 m）的约化 Hessian 做逐行 vjp；
* 0.05 m 上 `||F||∞ = 1.16e-7` 撞上绝对门禁 `1e-9`，优化器被迫中断（`panel/p1_cst_gradient.py:306-311`）；
* 原配点实验的尾缘梯度对网格很敏感。历史 ISSUE71 的固定环量/无切口证据已撤回，
  不能据此断言所有离散都不收敛；修正后的物理位置探针见 §10。

## 2. 理论

Dirichlet 总势公式里，壁面偶极强度 μ 与尾迹偶极强度 Γ 满足

```
A μ + W Γ = b            （壁面：总势迹积分方程 / Galerkin 弱形式）
C μ − I Γ = 0            （后缘：Δμ_TE = Γ，即势跳变等于尾迹强度）
```

第二行就是 Morino 的尾迹条件：定常尾迹沿流方向强度不变，其值等于后缘势跳变。这一行在
`panel/p1_lifting_g1.py:465-477` 已经装配成 `J`（实测 `J` 的 Γ 块是 `−I`，`J` 行形如
`+μ_u −μ_l −Γ = 0`）。

于是方程组本身是**欠定**的：未知量 `nt + ng`，方程 `nt + ng`（`A` 是方阵），但 `J` 只确定
「Γ 等于势跳变」，没有确定 Γ 取多少。两种经典收口方式：

1. **线性闭包**（A502/PANAIR 式）：直接解方阵 `[A W; J]`，Γ 完全由壁面方程 + 势跳变决定，
   不再附加条件。便宜、无迭代、无奇异点，但后缘 Δp 只剩离散精度，实测并不小（见 §3.1）。
2. **迭代 Kutta**（PMARC 的 ITKUT 式）：先由线性闭包得到 Γ 的初值，再迭代调整 Γ，使
   Δp_TE → 0。外层只有 ng 个未知量。

当前生产代码其实是第 2 类的变体，但把「调整 Γ」和「求解 μ」耦合进同一个全状态最小二乘里，
所以状态维数高、Hessian 大、且用权重 25 软约束 Δp 而不是精确求解。

## 3. 实测（同网格 A/B，α=4°，GRASS_DEFAULTS，`include_ideal_drag=False`）

### 3.1 线性闭包 vs 生产压力 Kutta（`runs/scratch_opt/morino_probe.py`）

| 面数 | 线性 CL | 线性 \|Γ\|max | 线性 TE Δp rms | 线性壁面残差 | 生产 CL | 生产 TE Δp rms | 生产壁面残差 |
|---|---|---|---|---|---|---|---|
| 132 | 0.342223 | 0.1694 | 7.44e−2 | 9.9e−16 | 0.341870 | 1.0e−7 | 9.1e−4 |
| 156 | 0.346299 | 0.1703 | — | — | 0.342495 | — | — |
| 1720 | 0.349650 | 0.1818 | 3.29e−2 | 3.3e−15 | 0.348521 | 8.0e−14 | 1.6e−6 |
| 6640 | 0.348926 | 0.1829 | — | — | 0.348620 | — | — |

结论：线性闭包把壁面方程解到机器精度，但**后缘 Δp 留着 0.03~0.07**（约 3~7% 动压），
不能当作满足 Kutta 的解；生产解则把 Δp 压到 1e−13，代价是壁面残差 1e−6 和全状态非线性求解。
两者在细网格上 CL 只差 0.08~0.3%，粗网格差到 1.2%。

### 3.2 两级 ITKUT（`runs/scratch_opt/itkut_probe.py`）

做法：先由线性闭包得到 Γ⁰；令 `μ(Γ)` 为「给定 Γ 时壁面方程的最小二乘解」，它对 Γ 是**仿射**的，
所以可以一次 `lstsq` 求出 `μ = μ_base + M Γ`（`ng+1` 个右端项一次算完）；然后对 ng 维的
`p(Γ) = Δp_TE(μ(Γ), Γ)` 做 Newton。

| 网格 | 迭代 | \|p\| rms | CL | 壁面 relres |
|---|---|---|---|---|
| 132 面（ng=9） | 0 | 7.44e−2 | 0.342223 | 4.2e−16 |
| | 1 | 6.89e−4 | 0.353875 | 7.45e−4 |
| | 2 | 2.52e−9 | 0.353547 | 7.45e−4 |
| | 3 | 8.2e−16 | 0.353547 | 7.45e−4 |
| 1720 面（ng=41） | 0 | 3.29e−2 | 0.349650 | 7.4e−15 |
| | 1 | 1.74e−5 | 0.348509 | 9.68e−7 |
| | 2 | 7.87e−12 | 0.348508 | 9.69e−7 |
| | 3 | 9.9e−15 | 0.348508 | 9.69e−7 |
| 生产（1720） | — | 7.65e−14 | 0.348521 | 9.60e−7 |

**两次 Newton 就收敛到机器精度**，单次外层迭代只解一个 ng 维线性系统。

### 3.3 关键发现：粗网格上 Kutta 根不唯一

* 1720 面：ITKUT 与生产解 CL 差 1.3e−5、Γ 差 8e−5 —— 两条路径收敛到同一个解。
* 132 面：ITKUT 给 0.353547，生产给 0.341870，差 3.4%，两者 Δp 都到 1e−13 以下、壁面残差同量级。

也就是说，在 132 面这种粗网格上，Δp_TE(Γ)=0 有**多个根**，两个算法只是在同一个多根问题里
各自收敛到不同的根。根因还是 §1 第三条：粗网格后缘速度场不具备分辨 Γ 的能力。**这不是换算法
能修的，必须靠网格/高阶/消奇异。**

## 4. 方案

### 4.1 建议实现（`kutta='potential_jump'`，默认仍为 pressure）

1. 装配 `A, W, b, J`（已有）后，先解一次 `[B; J]` 得到 Γ⁰（线性闭包），作为起点。
2. 一次 `lstsq` 求出仿射映射 `μ = μ_base + M Γ`（`ng+1` 个右端项），之后每次评估只需 `M @ Γ`。
3. 对 `p(Γ)` 做 Newton（Jacobian 由 `grad` 链式得到，或 1 次 VJP），2~3 步到 1e−13。
4. 隐式求导降到 ng 维：`H = ∂p/∂Γ`（ng×ng），伴随解一次 `Hᵀλ = ∂J/∂Γ`，再对 xyz/cst 做一次 VJP。
   0.05 m 上 Hessian 从 3403² 降到 81²。

### 4.2 验收判据（尺度无关，替换绝对 1e−9）

```
rel_grad = ||F||∞ / (||B||∞ · ||s||∞ + ||rhs||∞)   ≤ 1e-10
rel_wall = ||B s − rhs|| / ||rhs||                  ≤ 1e-8
|p| rms / q_ref                                     ≤ 1e-10
```

三条同时比，另加「Γ 的 Newton 步长相对变化 ≤ 1e−10」作为收敛证据。绝对阈值只在单一网格上
有意义，这正是 `ISSUE84` 的教训。

实测（`runs/scratch_opt/scale_check.py`，`rel_inf = ||F||∞ / (||B||_F ||s|| + ||rhs||)`）：

| 网格 | \|F\|∞ | rel_inf |
|---|---|---|
| 132 面 | 1.96e−11 | 2.0e−12 |
| 1720 面 | 6.29e−11 | 2.7e−12 |
| 6640 面 | 3.28e−7 | 7.9e−9 |

按 ||F||/(||B|| ||s||+||rhs||) 归一后 0.05 m 仍有 7.9e−9，比其他网格差三个量级，看上去像求解器
真的停在了不平稳点。**但这仍然是错的判据**：分母里的 `||B|| ||s||` 与残差的量纲并不匹配，
真正该比的是「再走一个 Newton 步还能移动多少」，见 §6.1。改用 Newton 递减量之后 0.05 m 的
相对步长只有 1.2e−12，说明它早就是平稳点，之前的 1e−9 只是单位错了。

### 4.3 风险

* 换闭包会改变**粗网格**数值（最大 3.4%），所有冻结样例与文档数字需要重跑，不能悄悄替换默认。
* 多根问题在粗网格上没有「正确答案」可选，只能通过收敛性研究（多层网格 + 外推）判定。
* 0.05 m 上仿射映射本身要一次 3403×3322 的 `lstsq`（约 30 s），比现在的逐行 Hessian 便宜两个量级，
  但仍不是免费的。

## 5. 未做 / 待验证

* 0.05 m（6640 面）上 ITKUT 的收敛性与 CL —— 正在跑（`runs/scratch_opt/itkut_005.log`）。
* 用 3~4 层网格做收敛性外推，确定哪条闭包/哪个根在细网格极限下正确。
* 与外部代码（VSPAERO）同几何对比：本机没有 VSPAERO 源码，`D:/3D/Projects/OpenVSP-3.55.0-win64` 目录不可读。
* 圆润后缘 / 边缘富化的对照实验（`ISSUE71` §6.3 的下一步）。

## 6. 已实现（2026-11-07）

### 6.1 尺度无关的平稳性判据（第 1 步）

`panel/p1_cst_gradient.py` 原来的门禁是绝对量 `||F||∞ ≤ 1e-9`，现在是 Newton 递减量：

```python
factor, hessian_status = torch.linalg.cholesky_ex(hessian)
step = torch.cholesky_solve(stationarity[:, None], factor)[:, 0]
relative_step = ||step||∞ / ||z||∞          # 必须 ≤ 1e-8
```

实测（`runs/scratch_opt/decrement_probe.py`）：

| 网格 | \|F\|∞ | \|H\|_F | \|dz\|∞/\|z\|∞ |
|---|---|---|---|
| 132 面 | 1.96e−11 | 8.95e3 | 2.26e−9 |
| 1720 面 | 6.29e−11 | 2.41e6 | 2.84e−14 |
| 6640 面 | 3.28e−7 | 4.57e7 | 1.25e−12 |

0.05 m 的 `|F|∞` 比 0.1 m 大 5000 倍，但 H 大 19 倍、解大 2 倍，剩下的 Newton 步只有 1.2e−12。
改判据后 0.05 m 跑通（`runs/scratch_opt/gate_check.py`）：

```
surface_0.05.json faces=6640 CL=0.348620308 CD=0.009278360 rel_step=1.087e-12  192 s
```

回归：132/156 面的 CL/CD/CDi 与 `runs/scratch_opt/opt02/implicit_ref.json` 仍逐位一致（12 位小数），
CST 梯度相对差 6.9e−9 / 8.9e−9，与改前相同。

### 6.2 Morino 势跳变闭包（第 2 步）

`fixed_wake_geometry(..., kutta="potential_jump")` 走 `_PotentialJumpImplicit`：

1. `_affine_state`：一次 `lstsq`（ng+1 个右端项）把壁面解写成 `mu(gamma) = mu_base + M gamma`；
2. `_potential_jump_newton`：对 `p(gamma)=0` 做 Newton，Jacobian 用显式的
   `dp/dgamma = Jp @ M`（`Jp` 来自 `TorchPressureModel.jacobian`，不需要自动微分）；
3. 反向 ng 维伴随：`lambda = (dp/dgamma)^-T (dJ/dgamma)`，
   `dJ/dx = dJ/dx|_gamma - lambda^T dp/dx|_gamma`，全程一阶图，没有 ndof×ndof 的 Hessian。

梯度校验（中心差分 h=1e−6）。**2026-11-07 更正：这里的第一次校验是瞎的。**

原先的 `runs/scratch_opt/pj_grad_check.py` 对 `coef.sum()` 求导、差分也是 `c2.sum()`，
权重恰好全为 1，于是掩盖了反向里的一个真 bug：它把每个系数各自的伴随向量**不加权**地全部减掉
（`sum((l * p).sum() for l in lam)`），而不是按上游权重合并成一个伴随向量。只要目标是单个系数
（`dCL` 单独、或优化器里的 `dCDi`），梯度就会错几个百分点。修法是
`lam = Σ_k grad_coef[k] · solve(dpdg^T, target_k)`，`total = (coef·grad_coef).sum() − (lam·p).sum()`。

当时的错误幅度（3×3 CST 翼，`runs/scratch_opt/pj_xyz_fd.py`）：xyz(5,2) 的 CL 解析 1.094184
对差分 1.023763，CD 解析 +0.9615 对差分 −0.1752；而数值 IFT 恒等式
（`runs/scratch_opt/pj_ift_probe3.py`）给出 1.023763 / −0.175181，证明形式没错、实现错。

修好后逐系数校验（`runs/scratch_opt/pj_grad_check_percoef.py`，4 弦向 × 2 展向网格，
CL=0.290587122 CD=0.001793706 CDi=0.004122875）：

| 变量 | 分量 | 解析 | 差分 | 相对差 |
|---|---|---|---|---|
| cst[0,3] | CL | 2.06107877e−1 | 2.06107876e−1 | 5.9e−9 |
| cst[0,3] | CD | 1.33821109e−2 | 1.33821106e−2 | 2.2e−8 |
| cst[0,3] | CDi | 6.72496021e−3 | 6.72496027e−3 | 7.6e−9 |
| cst[1,16] | CL | −1.62430629e−2 | −1.62430652e−2 | 1.4e−7 |
| cst[1,16] | CDi | 1.93345135e−5 | 1.93344720e−5 | 2.1e−6 |
| cst[1,5] | CL | 4.30979504e−2 | 4.30979450e−2 | 1.3e−7 |

回归测试：`test/test_p1_cst_gradient.py::test_potential_jump_gradient_matches_central_difference`
（index 0/8/16，逐系数、逐分量）。教训：伴随实现里任何一个「全 1 权重」的校验都可能是瞎的，
校验必须按输出分量拆开做。

### 6.3 两种闭包的 A/B

`runs/scratch_opt/pj_ab.py`、`runs/scratch_opt/pj_fine.py`（α=4°，GRASS_DEFAULTS）：

| 尺寸 | 面数 | pressure CL | potential_jump CL | 相对差 | pressure 耗时 | pj 耗时 |
|---|---|---|---|---|---|---|
| 0.5 | 132 | 0.341870270 | 0.353547086 | 3.4% | 0.3 s | 0.1 s |
| 0.35 | 156 | 0.342494545 | 0.352629840 | 3.0% | 0.1 s | 0.1 s |
| 0.2 | 460 | 0.353311351 | 0.353280128 | 0.009% | 0.5 s | 0.4 s |
| 0.1 | 1720 | 0.348520615 | 0.348508182 | 0.004% | 68 s | **5.4 s** |
| 0.05 | 6640 | 0.348620308 | 0.348617456 | 0.0008% | 192 s | **85 s** |

potential_jump 在每个尺寸上 `|p|∞` 都到 1e−13~1e−15（pressure 在 1e−10~1e−11），而且省掉了
ndof 维 Hessian，0.05 m 上快一倍多。0.5/0.35 m 的 3% 差异不是实现错误，是 §3.3 的多根现象。

### 6.4 默认值

`kutta="pressure"` 仍是默认。三层网格的对比见 §7.1：到 0.05 m 两条闭包的 CL 差 8.5e−6，
比网格误差小一个量级，所以「哪条更对」在细网格上已经不需要争论；粗网格的 3% 是多根，不是对错。
升默认的唯一阻碍是冻结样例，见 §7.3。


## 7. 网格收敛与「能不能归档旧闭包」（2026-11-07）

### 7.1 三层网格

同一 GRASS 构型、α=4°，两条闭包各自的 CL：

| 尺寸 | 面数 | pressure | potential_jump | 差 |
|---|---|---|---|---|
| 0.2 | 460 | 0.353311351 | 0.353280128 | 3.1e−5 |
| 0.1 | 1720 | 0.348520615 | 0.348508182 | 1.2e−5 |
| 0.05 | 6640 | 0.348620308 | 0.348617456 | **8.5e−6** |

各自最后一步的增量：pressure 0.1→0.05 为 +9.97e−5，potential_jump 为 +1.09e−4。也就是
**两条闭包之间的差（8.5e−6）比网格本身误差（约 1e−4）小一个量级**，且随网格收紧。到 0.05 m，
「用哪条闭包」已经不影响 CL 的第三位有效数字；0.5/0.35 m 的 3% 是两个不同的根，不是闭包优劣。

### 7.2 归档评估

能归档的是**「把最后一格速度差直接当 Kutta 方程」这个做法本身**，但它在这份代码里有两个化身，
不能一起处理：

1. `panel/p1_cst_gradient.py` 的 `kutta="pressure"`（本文档的对照项）。它其实已经是混合式——硬约束
   `J`（势跳变）与 `K`（后缘速度跳变）只用来构造初值，真正的闭包是在 `J` 的零空间里对
   `|B s − rhs|² + 25|Δp|²` 做最小二乘。它可以被 `potential_jump` 取代，前提见下。
2. `panel/p1_galerkin_solver.py` 的 `kutta="signed_speed"`（主 P1 求解器、自由尾迹、耦合材料尾迹
   走的那条）。**这不是同一个东西**：`_PotentialJumpImplicit` 只覆盖固定尾迹的 CST 路径，替换它
   要另开一轮工作。

暂不归档的三个理由：

* **冻结样例**：`test/test_p1_cst_gradient.py` 的期望值、`verification/cst_gradient/README.md`、
  Web 冒烟与 SLSQP 记录都用 pressure。换默认会让这些数字一次性失效，必须先重跑并在文档注明
  「换过闭包」。
* **诊断价值**：pressure 是**超定**最小二乘（872 行 vs 487 未知），残差停在 1e−10 是真信号；
  potential_jump 是**方阵** Newton，`|p|→1e−15` 是构造出来的，它无法告诉你网格好坏。两条一起跑
  才能区分「离散不够」和「闭包选错」。
* **稳健性**：Newton 没有回溯线搜索，坏几何（尖前缘、大上反、强后掠）下会失败；现在失败会
  `raise RuntimeError`（2026-11-07 加的守卫），但没有 fallback。pressure 的 `max_nfev` + 罚项
  路径在这种情形下更耐操。

## 8. 尖端后缘掠角与闭包条件数（2026-11-08）

### 8.1 现象

椭圆平面形按 θ 均匀截到 0.97·π/2（13 段、chord_panels 8、每段 8 个展向面板、6176 面）时，
`potential_jump` 的 γ(y) 在全展向光滑，但在 +y 尖端 y≈3.90 处猛刺：γ 从 +0.5 跳到 −2.13，
全展向 12 次变号，|dγ/dy| 最大 181，Trefftz 展向效率 e=12.96。同网格 `pressure` 光滑
（2 次变号，自身 Kutta 残差 4.84e−14），矩形翼同密度（1056/2080/3104 面）也是 0 次变号。

网格质量不是原因：细网格 min_edge 3.6e−3、min_area 1.58e−5、max_aspect 15.0（粗网格 16.3）。

### 8.2 机制

尖端弦长 0.094、后缘与来流夹角 77°，尾迹在这最后 5 cm 内是 7 条 0.007 宽 × 20 弦长的条带
（纵横比约 2900）。闭包 Jacobian `dp/dγ` 里相邻的跳变行因此近共线，系统近奇异；Newton 仍然
收敛（残差 3e−15），但解里带着一个任意大小的近零方向分量。

诊断办法：monkeypatch `torch.linalg.solve`，对 pj Newton 的方阵取 `torch.linalg.svdvals`
算条件数，同时记录每步最大修正落在哪个 γ 索引。实测（同一 6176 面网格，只改尖端）：

| 尖端 | 后缘掠角 | cond(dp/dγ) | γ 变号 | CL |
|---|---|---|---|---|
| 矩形翼（96 个展向面板） | 0° | 1.1e2 | 0 | 0.276199 |
| 椭圆到 0.85 展长 + 线性收尖到 0.24 | 54° | 4.5e4 | 0 | 0.302000 |
| 椭圆到 0.90 展长 + 线性收尖到 0.12 | 62° | 7.1e5 | 0 | 0.303038 |
| 纯椭圆截到 0.97 展长 | 77° | 7.3e6 | 12（γ→−2.13） | 0.253478 |

结论：**残差小不等于解唯一**。这一档的 `stationarity_inf` 是 3e−15，如果只看残差门禁，
它会以「收敛」的姿态进入梯度与优化。

### 8.3 门禁

`panel/p1_potential_jump_torch.py`：

* `_condition(matrix)`：`svdvals` 首末之比，最小值为 0 时返回 inf；ng>2048 跳过（SVD 是 O(ng³)）。
* `potential_jump_newton` 返回 7 元组，末项是循环中见到的最大 condition。
* `solve_potential_jump(..., condition_limit=1e6)` 在残差门禁之外再查条件数，超限抛
  `RuntimeError("potential-jump Kutta is ill-conditioned: cond(dp/dgamma)=… exceeds …, so the
  trailing-edge circulation is not unique …")`，并把 `condition`/`condition_limit` 写进 diagnostics。
* `panel/p1_cst_gradient.py` 的 `_PotentialJumpImplicit.forward` 加同一道门禁：近奇异的
  状态上的隐式梯度同样是任意的。

阈值取 1e6 的依据就是 8.2 的表：放行 4.5e4 的有界尖端，拒掉 1.25e6 的 0.90 尖端
（它已有 2 次变号、|dγ/dy| 177）与 7.3e6 的纯 0.97 尖端。cond 不是完美预测器——
N11 c6 sps6 只有 6.4e4 却有 8 次变号，N13 c8 sps4 只有 6.8e2 却有 3 次——门禁只挡极端档，
小振幅的展向锯齿仍需靠 γ 变号/斜率这类诊断看。

### 8.4 对锚点的影响

椭圆锚点改用有界尖端（0.85 展长 + 线性收尖到弦 0.24，后缘掠角 54°）：1560 面 e=0.9265、
CL_alpha 4.4341（提升线 4.518），同展长同面积矩形 0.8678；细化到 6176 面 e=0.9884，趋近椭圆
最优。证据与断言在 `test/test_physics_anchors.py`，门禁用例用最便宜的复现（9 段 c6 sps8、
3096 面、cond 1.63e6）。

### 8.5 pressure 为什么没有这个模态

两条闭包求解的是同一个 γ，但方程结构不同：

* `potential_jump` 的行是「每个后缘站位一条：势跳变 = 0」，**方阵**，γ 只由后缘数据决定。
  近零方向就是「让每一站的后缘 Δp 都几乎不变的一整条展向 γ 分布」——它真实存在，因为尖端
  后缘面只有 0.007 宽，那里 Δp 对 Γ 的灵敏度本来就小。
* `pressure` 解的是超定最小二乘 `[B; 25·K] z = [b; 0]`：γ 主要由**整片壁面方程** B 安放，
  后缘 Δp 只是权重 25 的罚项，从不被求逆。同一条 γ 近零方向在 B 里仍有残差，所以它不会
  变成自由方向。

实测（monkeypatch `torch.linalg.solve` 抓最后一轮 dp/dγ，取最小奇异值的右奇异向量）：

| 网格 | 面数 | cond | 近零方向的质量分布 |
|---|---|---|---|
| 纯椭圆 9 段 c6 sps8 | 3096 | 1.63e6 | 97.2% 集中在 2 个站位（y=3.888/3.901/3.915），90% 只需 2 站 |
| 纯椭圆 9 段 c6 sps6 | 2328 | 3.98e3 | top3 共 73.9%，90% 需要 5 站 |

同一张 3096 面网格上 `pressure` 的 γ 光滑（2 次变号、自身 Kutta 残差 4.84e−14、
CL=0.280094，有界尖端同族网格 0.302/0.303）。**但这不是说 pressure 对**：第 3、7 节已经
证明它在弦向/展向细化上收敛到错根。两者的坏法不同——pj 静默交出一个**非唯一**的解，
pressure 静默交出一个**收敛到错根**的 CL。门禁只能挡住前者。

### 8.6 网格层处理：后缘条带体检（2026-11-09）

§8.3 的门禁只能在求解时拒绝，不能事先告诉用户「先把网格改对」。新增 `panel/te_strip_audit.py`：
`te_strip_diagnostics(wing)` 对每条展向后缘条带给出宽度、TE 掠角、弦长和尾迹条纹纵横比，
按三条判据生成 `warnings`（每种最多 3 条，`warning_counts` 记总数）与 `ok`：

| 判据 | 阈值 | 依据 |
|---|---|---|
| TE 掠角 | > 60° | 同一平面形上 53.8° → 4.5e4、62.2° → 7.1e5、74.0° → 1.6e6 |
| 尖端/根部弦长 | < 0.03 | 0.97 截断 = 0.0706（可接受），0.99999 = 0.0244（共尖） |
| 条带纵横比 | > 4000 | 仅兜底：有界尖端细化到 sps8 时 1613，cond 仍 4.5e4 |

关键取舍：**纵横比本身不是判据**（它主要反映分辨率），设成 1000 会把健康的细化网格判红；
真正的门是掠角与尖端弦长。阈值来自 collocation 闭包，Galerkin 壁面好约两个数量级，所以
`ok=False` 是「先查一下」，不是「一定错」。

同一组锚点数字：`_sections(9, 0.97, 0.0942)` + sps8 → 掠角 73.6°、`ok=False`，和 §8.3 门禁
拒绝的是同一张 3096 面网格；`_sections(9, 0.85, 0.24)` + sps8 → 53.8°、`ok=True`（纵横比
1613 但被放行）。`bounded_elliptic_sections()` 直接给出合格的椭圆平面形（默认 0.85 展长 +
线性收尖到 0.12·根弦，53.8°）。命令行：`python -m tools.te_strip_audit --demo cusp`，或
`--mesh <build_grass_mesh 写出的 JSON>`；`ok=False` 时退出码 1，脚本可在优化前拒绝网格。

限制：只报告、不改网格（尖端要在截面里切）；阈值是经验值；cond 不是完美预测器（N11 c6 sps6
cond 6.4e4 却有 8 次变号）；尖端弦长比用后缘点的 `hypot(x, z)` 而不是截面弦长，厚翼会小几个
百分点，所以 0.03 是松阈值。

### 7.3 建议顺序（2026-11-07 修订）

第 1 步原计划「直接升默认」，执行到一半被 §7.4 的反证否决，顺序改成：

1. 把势跳变闭包移植进**生产求解器**（`panel/p1_lifting_g2.py:315 solve_multi`）之后，才谈升默认；
   在那之前 `fixed_wake_cst` / `fixed_wake_sections` 保留 `kutta=` 参数、默认 `pressure`；
2. `ISSUE84` 的 `pressure_kutta_acceptance()` 在文档里标注为 pressure 专用，不再当通用判据；
3. 主求解器的 `signed_speed` 另行评估——它是 §7.2 第 2 条，与本结论不共享代码。

### 7.4 为什么默认仍然是 pressure（升默认被否决）

`kutta="potential_jump"` 是 2026-11-07 新增的 opt-in 路径，`pressure` 仍是默认。
原因不是数值，而是一条验收性质：

* `panel/p1_lifting_g2.py:315`：`solve_multi(mesh, alpha_deg=5., *, kutta='pressure', ...)`
  —— 生产 P1 求解器的默认闭包**也是 pressure**。
* `test/test_p1_cst_gradient.py::test_cst_weak_operators_match_production` 断言可微路径与
  生产求解器的 CL/CD 差 ≤2e−9。
* `runs/scratch_opt/production_vs_closures.py`（`cst_geometry(code, 3, 3)`，
  `code=[.17]*8+[-.17]*8+[0,0]`，α=4°）实测：production CL=0.207708175686、
  CD=−0.005696693980；pressure 差 **−4.41e−15 / −1.73e−16**；potential_jump
  CL=0.207419222138（差 −2.89e−4）、CD 差 −1.70e−5。

所以「升默认」不是改一个 kwarg，而是要把 pj 移植进 `solve_multi` / `p1_lifting_g2` 整条线，
并重跑生产验收。在那之前把默认换掉会直接破坏「可微路径 = 生产 P1」这条性质。

### 7.5 移植完成（2026-11-07）

上面第 1 步已完成：生产求解器现在也支持 `kutta="potential_jump"`，默认仍是 `pressure`。

* 新增 `panel/p1_potential_jump_torch.py`：`affine_state`、`potential_jump_newton`、
  `solve_potential_jump` 是**唯一实现**，生产路径（`panel/p1_solve_torch.py`）与可微路径
  （`panel/p1_cst_gradient.py` 的 `_FixedWakeImplicit` / `_PotentialJumpImplicit`）都调它，
  免得两份 Newton 各自漂移。该模块只依赖 torch 和 `p1_rank_aware_torch`，不破坏
  「`p1_solve_torch` 不 import numpy/scipy」这条性质。
* `solve_multi_torch` 的 `kutta` 接受三个值；pj 分支在 J 精确成立的仿射族上对 TE 压差做 Newton，
  不收敛就 `raise RuntimeError`（带 evaluations / stationarity_inf / relative / tolerance），
  不返回「看起来像解」的状态。`nonlinear` 里给出 `method="morino_potential_jump"`。
* 对齐证据（`runs/scratch_opt/pj_production_port.py`、`pj_port_grass.py`、`pj_port_multi.py`）：
  - 3×3 CST 翼：`solve_multi(kutta="potential_jump")` 得 CL=0.207419222138、CD=−0.005713680157，
    与 `fixed_wake_cst(..., kutta="potential_jump")` 差 **6.6e−15 / 6.6e−17**；同一翼的 pressure
    路径与生产的差仍是 4.41e−15。
  - GRASS 两截面 80 面：pressure CL=0.290700055、pj CL=0.290587122（差 1.13e−4，粗网格多根），
    生产 pj 与可微 pj 差 1.7e−15。
  - 双翼（2 个 group、ng=6）：pj 两次 Newton 收敛，`jump_rms=0`（仿射映射构造上成立）、
    `pressure_kutta_rms=2.63e−15`，CL=0.294953687 对 pressure 的 0.295070972。
* 回归：`test/test_p1_cst_gradient.py` 新增 `test_potential_jump_matches_production_solver` 与
  `test_potential_jump_production_handles_two_groups`；三个 P1 文件 20 passed。
* **默认为什么不跟着换**：`docs/issue92/references/` 与大部分冻结测试的期望值都是 pressure 下的数，
  换默认要让它们一次性失效并重跑生成参考。代码上现在只剩「改一个默认值 + 重跑参考」这一步，
  这是策略决定，不是技术阻塞。

## 9. 升力线、势跳变与压力闭包复核（2026-11-09）

本节记录修复前的代码阅读与现场复算；当时未修改求解器、升力线实现或默认设置，后续修复见 §10。环境为 Windows x64、
项目 myml Python 3.13，torch CPU 单线程，BLAS/OpenMP/MKL 线程数均设为 1。
临时脚本 `runs/scratch/lifting_line_closure_audit.py`，数据
`runs/verification/lifting_line_closure/audit.json`；两者按仓库规则不提交。
复现：在上述线程环境下执行
`C:/Users/zyx20/anaconda3/envs/myml/python.exe runs/scratch/lifting_line_closure_audit.py`。

### 9.1 先纠正比较量

`tools/lifting_line.py` 的矩阵、`CL = pi AR A1`、`CDi = pi AR sum(n An²)`、
`e = A1² / sum(n An²)` 彼此一致。但此处 `b` 已定义为全翼展，因此环量应为
`Gamma = 2 b U_inf sum(An sin(n theta))`，代码第 116 行的 `4 b` 多出一倍。
由 `CL_Gamma = 2 integral(Gamma dy)/(U_inf S)` 积分即可独立核算。
当前函数没有速度参数，可返回 `Gamma/U_inf` 并明确单位约定。

现场用精确椭圆弦长函数、b=8、根弦=2、alpha=4°：
模态 CL=0.314964608，返回 gamma 积分所得 CL=0.629864974，比例 1.999796；
小于精确 2 的差来自输出配点上的梯形积分。精确椭圆的 e=1，斜率为 4.511535683；
它不是此前有界尖端分段线性平面形的 e=0.9994、斜率 4.5213。

还有两个独立的接口/展示问题：

* 函数文档称 gamma 对应输入 y，实际总是返回 `2*terms+1` 个内部配点上的值，
  还未返回这些配点；输入 20001 个 y 时默认输出只有 127 个 gamma。修正应返回配点
  坐标，或在输入 y 上重新计算正弦展开，并分别说明两种输出。
* `runs/scratch_opt/ll_compare.py` 与 `test/test_physics_anchors.py` 把
  `CDi/(CL²/(pi AR))` 标成 e，实际是 `1/e`。标准效率应为
  `CL²/(pi AR CDi)`。旧展示 61.661 对应标准 e 约 0.01622，4.761 对应约 0.21004。
  这两个例子仍异常，但不能再解释为效率暴涨。旧椭圆/矩形效率排序断言也须重新审查，
  并同时报告 CL_pressure、CL_Gamma，避免用不一致载荷归一化掩盖问题。

当前 `pytest -q test/test_lifting_line_reference.py -p no:cacheprovider` 为
4 passed；测试未覆盖 gamma 积分恒等式、输出坐标和跨模块效率定义，故未发现这些错误。

### 9.2 奇次展开与配点秩

仅取奇次项适用于对称平面形与对称有效攻角；它不会天然导致两条闭包差异。
但在全展向使用对称 theta 配点时，
`sin(n(pi-theta)) = sin(n theta)`（n 为奇数），矩阵的镜像行完全相同。
如果 N 个奇次未知量仅放 N 个全展向配点，秩至多 ceil(N/2)，不能只解释为尖端
`1/sin(theta)` 太大。当前 `2N+1` 个配点避免了独立方程数量不足。

用当前配点约定、矩形 AR=6 实测：

| 奇次项数 N | N 个全展向配点的秩 / 条件数 | 2N+1 个配点的秩 / 条件数 |
| --- | --- | --- |
| 31 | 16 / 6.57e16 | 31 / 3.67e2 |
| 63 | 32 / 1.41e17 | 63 / 1.51e3 |
| 127 | 64 / 8.29e17 | 127 / 6.15e3 |

这不表示所有方阵配点都不好：只在独立半翼上布置 N 个适当配点也可。
矩形翼的升力线 Gamma 在翼尖趋零，并非文档所说在侧边跳跃不连续；
边缘梯度和高阶模态收敛应另作分析。输入代码还只检查展向端点居中，
没有检查 chord(y)=chord(-y)，所以使用奇次项的对称性目前依赖调用者。

### 9.3 两条闭包实际上解什么

令 x=[mu; Gamma]，B 为加权壁面算子，Jx=0 表示
`mu_upper-mu_lower-Gamma=0`。两条路径都精确施加 J，都使用同一个
`TorchPressureModel`：
`p(x)=mean_top |grad(mu)|²-mean_bottom |grad(mu)|²`。
这里 p 是速度平方差，零点对应等压条件，数值本身不是有量纲压力差。

`pressure` 求：

```
min_x  ||Bx-b||² + w² ||p(x)||²,  subject to Jx=0
```

当前 w=5，即目标函数的系数是 25；不能写成残差行乘 25。

`potential_jump` 求：

```
mu*(Gamma) = argmin_mu ||B[mu;Gamma]-b||²
             subject to J[mu;Gamma]=0
mu*(Gamma) = mu_base + M Gamma
p([mu*(Gamma);Gamma]) = 0
```

外层 Newton 使用 `dp/dGamma = p_mu M + p_Gamma`；当前模型显式 p_Gamma=0，
但全导数 p_mu M 一般不为零。不能用显式 Gamma 列为零推断环量不可辨识。
对壁面最小二乘问题的条件最优性与全状态罚项目标的条件最优性不同，
所以这不是同一离散方程的两种等价算法。即便罚项权重趋于无穷，
也不能未经证明就把它们当作等价。

### 9.4 同网格实测：小势值改动可大幅改变后缘梯度

矩形 AR=16、厚度 0.12、alpha=2°、64 个展向单元、wake_length=5、
wake_panels=2、collocation；每个弦向分辨率只装配一次算子，供两条路径复用。
这是当前代码现场数据，不覆盖前文不同历史状态下的数字。
同平面形薄翼升力线给 CL=0.188754692；它仅是低阶参考，不是有限厚度真值。

| 弦向单元 | pressure CL_pressure / CL_Gamma | potential_jump CL_pressure / CL_Gamma |
| --- | --- | --- |
| 4 | 0.189956168 / 0.171737481 | 0.209379674 / 0.189627728 |
| 8 | 0.162878910 / 0.158703440 | 0.196501801 / 0.191458660 |
| 16 | 0.074809363 / 0.075167754 | 0.189503701 / 0.189461108 |

pressure 自身 p RMS 为 2.79e-7、8.47e-9、2.95e-9。保持它的 Gamma 不变，
把 mu 投影回上述壁面条件最小二乘族后，相对 mu 改变量仅
0.001116、0.000998、0.001508，p RMS 却变为 0.0366、0.1724、1.3412。
投影同时减小壁面目标值，因此这确实是 pressure 离开壁面条件最优族的量，
不是另一个环量或另一个输入造成的差异。pj 的对应偏离为 4.24e-14、7.45e-14、0。

这支持的机制是：全状态压力罚项可以利用小幅势值改变来调整高敏感的表面梯度，
后缘速度平方差很小仍不足以确定正确环量。它不证明具体边缘奇异机制已完全定位。
本组矩形 pj 雅可比条件数约 64、5.6、10.8，没有椭圆尖端的近奇异模态；
pressure 的弦向失效不能统一归因于尖端条件数。

也不能单凭两条路径 CL 不同、p 都小，就宣称「同一个方程有两个根」：
要验证该结论，必须把候选解放回同一个 mu*(Gamma) 族，再检查 p 和根的延拓。
前文 §3.3 与 §8.5 的多根/错根表述应按此限制理解。

### 9.5 修正与验收顺序

1. 先修 gamma 的系数与输出坐标、所有 e/1/e 标签及物理断言，增加环量积分与模态
   CL/CDi 一致性检查。保留旧输出，明确它们是旧定义，不按正确 e 混用。
2. 比较时固定同一几何、面积、攻角、尾迹长度、壁面离散与算子；明确区分
   CL_pressure 与 CL_Gamma（同一状态的后处理），以及 pressure/pj（两条求解路径）。
   先用 2pi 作薄翼锚点，再用独立二维计算获得同翼型、同工况的 a0；
   厚度可能解释部分差异，但不能据此认定已解释全部 5% 偏差。
3. 用固定 Gamma 的问题单独检验壁面势值、表面速度、Cp 的弦向/展向收敛，
   然后检验闭包。历史 ISSUE71 的固定环量与无切口负例仍需在候选修正上重做；
   只跳过最后一格、加权重、加迭代次数不是已有证据支持的解决办法。
4. 后缘压力恢复应使用上/下表面独立、位置匹配的收敛离散；可比较面积加权恢复、
   固定物理距离的多点外推、边缘加密/奇异富化或高阶迹空间。它们均是待验证方案，
   不能仅以残差更小或更贴升力线选型。有限夹角后缘尤其不能直接假设上下速度向量相等。
5. 保留环量约化路径作为诊断基线，但它仍使用相同压力模型，并非根治。
   用小攻角延拓、多初值、尺度归一后的奇异值/模态和 Gamma 平滑性检验分支与条件；
   尖端局部基函数/网格与尾迹匹配需独立修正。正则化或降阶若改变模型必须记录，
   不得通过裁剪 Gamma 隐藏振荡。
6. 最后验收 CL_pressure 与 CL_Gamma 的网格趋势、二维斜率极限、对称性、
   Trefftz CDi 和标准 e。固定 Gamma 下场量收敛、实际网格误差与物理参考通过之前，
   不以默认切换、冻结参考重写或条件数门禁宣称物理精度已解决。

## 10. 四步修复与验证（2026-11-09）

工作在用户指定的本地 master 完成，未检查 GitHub。环境：Windows x64，
`C:/Users/zyx20/anaconda3/envs/myml/python.exe`，torch CPU，
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`，torch 单线程。
以下严格区分代数收敛、网格趋势和物理精度；`physical_accuracy_accepted=false`。

### 10.1 比较定义已修正

`tools/lifting_line.py` 现在返回全翼展约定的
`Gamma/U_inf = 2*b*sum(A_n*sin(n*theta))`，样本输入保持原 y，函数输入返回含两端点的
余弦站位；返回 `y`、`modes`、`coefficients`，两翼尖 gamma 严格为零。奇次展开检查弦长
对称性；保留 `2N+1` 配点，原因是镜像行相同造成的秩限制，不再误称单纯尖端病态。

物理锚点与新对照脚本统一采用
`e=CL_Gamma^2/(pi*AR*CDi)`，CDi 调用已有精确分段线性 Trefftz 能量核。
`CL_pressure` 另报；旧 `1/e` 标签与旧中点阻力近似的数字不能直接混入。
独立 Gauss 积分检查环量/下洗的 CL、CDi 与模态公式，避免仅用同一错误公式自检。
椭圆/同面积矩形粗网格的修正 e 为 `0.978154/0.965248`；
撤销没有理论依据的效率差必须大于 0.03 的断言，保留正效率、上界与物理排序。

可复现的平面形比较：

```text
python -m tools.lifting_line_compare --case all --wall collocation --out runs/verification/lifting_line_fix/planforms
python -m tools.lifting_line_compare --case all --wall galerkin --recovery area --out runs/verification/lifting_line_fix/planforms_galerkin_area
python -m tools.lifting_line_compare --case all --wall galerkin --recovery area --kutta pressure --out runs/verification/lifting_line_fix/planforms_galerkin_area_pressure
```

脚本默认壁面与生产一致使用 Galerkin，`--wall collocation` 是显式控制。
保持相同几何、参考面积与环量，并同时保存 LL 与面元 gamma 的坐标/数据。
薄翼 `a0=2*pi` 仅作为低阶锚点；没有从三维答案反拟合厚度修正。

同为9截面、c6、每段4展向单元、1560面的 Galerkin+area 新对照：

| 平面形 | pressure斜率 / pj斜率 / LL斜率（每弧度） | pressure e / pj e / LL e |
| --- | --- | --- |
| 矩形 | 4.433424 / 4.433198 / 4.391205 | .981178 / .980970 / .960033 |
| 梢根比0.5锥形 | 4.939792 / 4.938941 / 4.841239 | .994461 / .994451 / .985509 |
| 有界椭圆 | 4.294669 / 4.293789 / 4.521323 | .978283 / .978256 / .999371 |

pressure 由本轮新增命令补跑，其余设置与 pj 相同。两法相对 LL 的斜率差分别为
`+0.96%/+0.96%`、`+2.04%/+2.02%`、`-5.01%/-5.03%`；
两法之间的斜率差仅 `.0051%/.0172%/.0205%`。
AR4、c32 的两闭包 CLp 也仅差 `.00848%`，但相对 LL 仍为 `-7.69%/-7.70%`。
因此共同离散/模型偏差比本组闭包差异更值得继续排查；这不是厚度已解释误差的证据。
本轮物理参考是薄翼升力线与解析极限，尚未加入同几何的风洞或独立三维解作验收。

原 collocation+face 控制的对应 e 为 `.120023/.946644/.978154`。
这同时改变了壁面和恢复，不能把全部改善归于恢复；受控单因素比较见 §10.3。
有界椭圆的升力斜率仍偏低，且未做完整二维截面斜率/几何网格验收。

### 10.2 严格固定环量，检验实际场量

`tools/lifting_line_study.py --case fixed-gamma` 使用共享 `affine_state`，
固定 `Gamma=.22*sqrt(1-(2y/b)^2)`，仅求条件最优壁面势值；
分别断言/记录 gamma 误差、Jx、条件最优偏离，不再使用错误的
`target + null_space(J) z`。探针在固定物理 x/y 上定位三角面，插值真实面上的 z 和势值；
速度/Cp 来自该面，避免最近重心替代真实位置。

```text
python -m tools.lifting_line_study --case fixed-gamma --out runs/verification/lifting_line_fix/fixed_gamma --chords 4 8 16 32 --span-panels 12
```

矩形 AR4、t/c=.12、alpha4°、wake10c、展向12单元。固定 gamma 误差为零，
`max|Jx|` 约 `3e-16`；`CL_Gamma=.3368181`、精确离散 e=.95944 不随弦向/壁面改变。

| 壁面 | c4 CLp | c8 CLp | c16 CLp | c32 CLp | 相邻网格探针最大速度变化 |
| --- | --- | --- | --- | --- | --- |
| collocation | .370901 | .342740 | .337985 | .335738 | .200914 / .276550 / .421445 |
| Galerkin | .357708 | .340007 | .333238 | .332859 | .190070 / .051305 / .040467 |

Galerkin 最大势值变化为 `.014726/.001496/.000445`；配点速度没有收敛趋势。
任意给定 gamma 不一定满足 Kutta，尖后缘本身可有奇异梯度，所以本实验以离开后缘的
固定物理探针为证据，不以 TE 梯度有界作为固定 gamma 问题的验收条件。

另执行 c16、展向 `8/16/24` 单元的 Galerkin 固定 gamma 对照（同一脚本，
`--chords 16 --walls galerkin --span-panels N`，输出 `span_sN`）。
最大势值变化 `.000351742/.000091566`，速度 `.00198258/.00201958`、
Cp `.00370926/.00406944`。势值改善，速度/Cp 尚不能声称一致收敛。
等比加密 `8/16/32` 的后一步为势值 `.000133728`、速度 `.00172390`、
Cp `.00449462`，仍不支持 Cp 在这些探针上收敛。
不同展向网格是同一解析 gamma 的不同分段线性表示，因此其积分升力/能量不要求逐值相等。

`tools/issue71/fixed_gamma.py` 的固定 gamma 求解已修复；旧物理探针仍为历史用途。
`tools/issue71/bisect_cut.py` 已停用：其旧固定 gamma 构造无效，
`nocut` 也从未真正移除切口/尾迹自由度。归档 ISSUE71 §2/§3 的结论已明确撤回。
这次没有声称重建了独立的无切口负例。

### 10.3 同位置、分岸恢复速度后再施加 Bernoulli

新增 `panel/p1_te_recovery.py`，上/下岸分别取同一实际 TE 节点上的入射面，
按面积恢复 P1 梯度，排除端盖与对岸；先重构向量，再计算速度平方差。
它与旧的“各岸邻面速度平方作算术平均”是不同离散，不能只修改标签。
对有限夹角后缘不强制上下速度向量相等。

生产与可微入口均支持 `pressure_recovery="area"`，pressure/pj 使用同一压力模型，
反向求导按当前几何重建恢复算子。测试包括分岸仿射切向梯度、规范平移、解析压力
Jacobian 与几何中心差分、两条闭包的生产/可微一致性与隐式梯度中心差分。
默认仍为 `face`，本轮没有用切换默认值掩盖控制算例。

```text
python -m tools.lifting_line_study --case closures --recovery face --out runs/verification/lifting_line_fix/closures_face
python -m tools.lifting_line_study --case closures --recovery area --out runs/verification/lifting_line_fix/closures_area
python -m tools.lifting_line_study --case closures --recovery quadratic --out runs/verification/lifting_line_fix/closures_quadratic
```

同上矩形参数、Galerkin 壁面：

| 弦向单元 | face pressure / pj CLp | area pressure / pj CLp |
| --- | --- | --- |
| 4 | .267136 / .267079 | .270346 / .270442 |
| 8 | .260166 / .260134 | .262877 / .262925 |
| 16 | .257074 / .257039 | .258954 / .258956 |
| 32 | .258493 / .258456 | .259611 / .259589 |

area 的 c16 两闭包 CLp 差约 `2e-6`，c32 差约 `2.2e-5`，
后者 CL_Gamma 为 `.262913/.262890`、e 为 `.951546/.951581`。
恢复后的载荷改善并不完全单调；pressure 与 pj 仍不是同一有限维问题。
配点+area 的 pressure CLp c4/8/16/32 为
`.272596/.131466/-.328977/-.879666`，恢复不能修复配点壁面的基础场量失效。

保留 `quadratic` 作为明确的研究控制，**未选作默认/推荐恢复**：
它用同岸邻域的二次势拟合与 QR，粗网格 pj 的 e 仅 `.04068`；
其 Galerkin c4/8/16/32 CLp 为 `.343124/.275300/.265868/.266418`。
多一个阶次不自动意味着较好，负结果保存在 `closures_quadratic/summary.json`。

### 10.4 Newton 尺度、修正量、条件数与分支

共享 Newton 增加 Armijo 回溯，按初值两岸速度平方归一化压力，
同时检查相对残差和实际 Newton 修正量；零残差初值仍检查 Jacobian，
包括最终根的条件数。报告迭代史、回溯数、最大原始条件数和最终行列归一后的条件数。
生产/可微入口均保留 `1e6` 原始条件数限制，不用归一后的较小数值绕过限制。
没有裁剪 gamma、人工置零或添加改变物理模型的正则化。

```text
python -m tools.lifting_line_branches --out runs/verification/lifting_line_fix/branches
```

脚本在同一个 `mu_base+M*gamma` 族上，用 signed、零、反号、5倍 signed 初值，
再做 alpha `-4/-2/0/2/4°` 的正向和反向延续；每个网格30次，c4/8/16合计90次。
全部通过代数判据，最大同角度 gamma 差 `7.58e-14`；
各网格最大条件数约 `1.91/2.01/2.08`，配对速度点积最小 `.850/.725/.621`，
两岸沿来流速度均为正。这支持本组分支稳定，**不是所有几何上唯一解的证明**。
报告 gamma 二阶差分、符号变化与反向流控制，不以“压差小”单独判断物理解。

零迎角仍有数值环量，最大约 c4 `.000314`、c8 `.000229`、
c16 `.000289`；c16有6次符号变化，而非零迎角均没有。
说明对称性/离散误差尚未消除，不能用良好条件数宣称物理正确。

针对性测试已实跑：

```text
python -m pytest -q test/test_lifting_line_reference.py test/test_trefftz_energy_exact.py test/test_physics_anchors.py -p no:cacheprovider
# 27 passed, 2 xfailed，86.04 s
python -m pytest -q test/test_lifting_line_studies.py test/test_p1_potential_jump_newton.py test/test_p1_te_recovery.py test/test_p1_fixed_gamma.py test/test_p1_pressure_kutta_torch_driver.py test/test_p1_cst_gradient.py -p no:cacheprovider
# 39 passed, 1 skipped，37.09 s
```

预期失败保留原配点压力闭包的 AR/弦向缺陷。新测试仅约束已验证的积分恒等式、
数值趋势和导数，不把当前三维数值当作实验真值。

全量 `python -m pytest -q test -p no:cacheprovider`：
**948 passed, 21 skipped, 2 xfailed，466.01 s**，日志
`runs/verification/lifting_line_fix/pytest_full.log`。
全量启动后补的单站位面积恢复及其相关可微路径另外实跑
`python -m pytest -q test/test_p1_te_recovery.py test/test_p1_cst_gradient.py -p no:cacheprovider`：
**21 passed，18.61 s**。
该单站位测试初版用了建网器不支持的两展向单元，报 ValueError；
改为已有合法网格上的单站位恢复夹具后通过。
开发失败回执保留于 `runs/verification/lifting_line_fix/development_failures.txt`，
不把测试夹具错误归于物理方法。
