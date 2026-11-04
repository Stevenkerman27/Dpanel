# 尾迹模型：表示、求值器、更新

状态：B2 耦合材料尾迹实验的架构说明
（`panel/p1_coupled_material_wake.py`）。本文不改变任何生产默认值：
`velocity_method='analytic_edges'` 仍为默认值，
`physical_accuracy_accepted` 仍为 false。

## 1. 三个相互独立的轴

| 轴 | 问题 | 由什么选择 | **不**改变什么 |
| --- | --- | --- | --- |
| 表示 | 尾迹*究竟*是什么对象？ | 尾迹模型选择（方法 1/2/3） | 壁面模型、Kutta 条件、求解 |
| 求值器 | 该对象的诱导速度如何计算？ | 仅方法 2 用 `area_quadrature`；任何方法都用求积阶数 / 扇形条带 / 近远场划分 | 尾迹面的几何、未知量、载荷 |
| 更新 | 尾迹几何在两次求解之间如何移动？ | B2 外循环（`iterations`、`relaxation`、`max_backtracks`、几何门禁、逐节点步长上限） | 固定几何下的速度模型 |

混淆这三者，正是此前“五个同级速度方法”表象的来源：
`projection_fan` 是方法 2 内部的面元积分器，不是一种尾迹
表示，改动 `fan_order` 改变的只是一个数值，而不是一个模型。

## 2. 方法 1 —— P1 偶极子面片，未正则化（`analytic_edges`）

默认方法与基准。尾迹是连续的 P1 偶极子面片：`mu` 在每个尾迹三角形上
线性变化，因此该面片携带两个物理贡献，

```
u(x) = 1/(4 pi) [ ∫_boundary  gamma_l(l) × (x - y) / |x-y|^3 dl
                 + ∫_area     gamma_f    × (x - y) / |x-y|^3 dA ]
gamma_l = linearly varying P1 trace on boundary / cut edges
gamma_f = grad_s(mu_f) × n_f     (in-panel distributed vorticity)
```

状态：生产默认，与 B2 完全耦合，精确求值（线性迹的边界积分为闭式，
面积项用求积）。它在面片**上**的取值是奇异的，因此在尾迹节点上取到的数
不是收敛的主值，不得作为主值报告。这一限制正是方法 2 存在的
原因，而不是方法 1 的缺陷。

## 3. 方法 2 —— 正则化 P1 面片（`rm_regularized_p1`）

相同的表示、相同的两个贡献，外加一个显式的 Rosenhead-Moore 核
`a`：每个分母都变为 `(|x-y|^2 + a^2)^(3/2)`，边界项与面积项均如此。
P1 面片**不会**被替换为等强度的边元涡丝。

状态：可选研究 / 生产候选；需要显式给出 `core_radius`。
`area_quadrature` 选择面积项的积分器：

| `area_quadrature` | 是什么 | 何时使用 |
| --- | --- | --- |
| `legacy_duffy` | 固定的张量积 Gauss-Legendre 加一次单顶点 Duffy 折叠 | 默认，最省；当核远小于面元时会产生混叠 |
| `projection_fan` | 绕最近点的投影扇形，逐面元条带 | 消除混叠的模式；用于精度证据的那个 |
| `projection_fan_adaptive` | 近处投影扇形、远处复合 Duffy、后验预算 | 可选，自门禁；报告自身的估计 |

`fan_order`、`fan_strips`、`fan_refinement_order`、`fan_refinement_strips`、
`near_ratio`、`adaptive_*` 和 `quadrature_order` 都是**本求值器的数值设置**。
它们从不选择模型，其中任何一个都不得被当作独立方法来呈现。

## 4. 方法 3 —— 拉格朗日涡丝（有文档，未接线）

一种不同的表示：携带环量的有向涡丝，诱导速度由 Biot-Savart / Rosenhead-Moore 线段求和给出，

```
u(x) = Σ_filaments Gamma/(4 pi) ∫ dl × (x - y) / (|x-y|^2 + a^2)^(3/2)
```

在能够耦合之前，它需要自己的环量传递、涡丝连通性、曲涡丝
自诱导处理以及 P1 到涡丝的映射。PR #83 的线段核是一个组件和一个
oracle，不是求解器。它被有意地**不**接受为 `velocity_method`；
`panel/wake_velocity_methods.py` 会以解释性消息拒绝它。

## 5. 弃用后端（`legacy_gauss`）

2026-10-30 之前的边 Gauss 求值器。仅保留以便记录了
`sampling_kernel='legacy_gauss'` 的冻结快照仍可重放；它不在
CLI 选项中出现，也不作为模型记录在文档中。

## 6. 数值控制与物理控制

| 控制项 | 类别 | 作用 |
| --- | --- | --- |
| `velocity_method` | 物理（模型） | 选择方法 1 或 2（3 保留） |
| `core_radius` | 物理（模型参数） | 设定方法 2 的 RM 核 |
| `area_quadrature` | 数值（方法 2 求值器） | 由哪个积分器计算面积项 |
| `quadrature_order` | 数值 | 旧边 Gauss 阶数 / Duffy 阶数；方法 1 忽略 |
| `fan_order`、`fan_strips` | 数值 | 投影扇形的分辨率 |
| `fan_refinement_*` | 数值（门禁） | 采样门禁用于对比的细化配置 |
| `near_ratio`、`adaptive_*` | 数值 | `projection_fan_adaptive` 的远近划分与预算 |
| 尾迹节点数（网格/几何） | 离散几何 | 面片在何处采样；**不是**速度设置 |
| `iterations`、`relaxation`、`max_backtracks` | 更新 | 混合逆/运动学映射如何移动面片 |

尾迹节点数与尾迹迭代次数是几何与 B2 循环的属性。它们通过改变离散化
与不动点来改变答案，因此必须在任何速度方法对比旁边一并报告；
`fan_order`/`fan_strips` 只改变固定几何下的求积误差。

## 7. 耦合了什么，没耦合什么

已耦合：壁面与尾迹共享 P1 势表示；壁面求解为
`solve_multi`（Dirichlet 势迹），压力-Kutta 条件将面片强度与壁面载荷联系起来，
B2 循环在几何门禁（根部/附着/面积/折叠/步长/壁面/
相交）与逐节点步长上限下，用运动学映射移动尾迹。门禁语义见
`docs/archive/issue/ISSUE84_T5_CONTACT_AND_STEP_CONTROL.md`。

未耦合：方法 3；任何基于涡丝的几何更新；Galerkin 壁面
弱形式（在 issue #71 中已实现并测量，但尚未成为 B2 壁面
求解器）；任何形式的物理精度声明。

## 8. API 与兼容性

* `velocity_method` 接受 `analytic_edges`（默认）与 `rm_regularized_p1`；
  `legacy_gauss` 仅为了重放旧快照而接受。
* `area_quadrature` 仅在方法 2 下有效，且仅限
  `legacy_duffy`、`projection_fan` 或 `projection_fan_adaptive`；在方法 1 下
  请求扇形会抛错，而不是被静默忽略。
* `python -m panel.run_p1_coupled_material_wake --describe-velocity-methods`
  打印下面的模型表。
* `panel/wake_velocity_methods.py` 是模型列表、状态、数值控制说明
  与错误消息的唯一事实来源；CLI 选项与求解器守卫都读取它。

```
method                 index  status                            coupled
legacy_gauss               0  deprecated, replay compatibility only False
analytic_edges             1  reference / production default   True
rm_regularized_p1          2  opt-in research / production candidate True
lagrangian_filaments       3  future model, not coupled        False
```

测试：`test/test_wake_velocity_methods.py`（注册表、fail-closed 守卫、CLI
表、文档契约），外加已有的 `test_p1_coupled_material_wake.py`、
`test_p1_regularized_sheet_velocity.py`、`test_p1_fan_quadrature_gate.py`、
`test_p1_adaptive_area.py`。

## 附录：术语对照（测试契约用词）

测试 `test/test_wake_velocity_methods.py::test_doc_page_covers_the_required_distinctions`
按英文关键词检查本页，因此中文化后这些词仍需出现；下表是唯一需要保留英文的地方。

| 中文 | 关键词（测试用） |
| --- | --- |
| 表示（三种尾迹模型） | representation，Method 1，Method 2，Method 3 |
| 求值器 | evaluator |
| 更新 | update |
| Rosenhead–Moore 正则化 | Rosenhead-Moore |
| Biot–Savart 求和 | Biot-Savart |
| 扇形阶数、扇形条带 | fan_order，fan_strips |
| 尾迹节点数、尾迹迭代数 | wake node count，wake iteration count |
| 物理精度是否被接受 | physical_accuracy_accepted |
| 未耦合 | not coupled |
| 投影扇形求积器 | projection_fan |
| 面积求积 | area_quadrature |
