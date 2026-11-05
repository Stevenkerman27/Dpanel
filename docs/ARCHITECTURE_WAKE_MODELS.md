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

## 9. G2 T5 物质前驱图

PR #110 的审计不改变图或尾迹更新。其原始 G2 T5 CAD 算例的完整导出为
[`verification/wake_material_graph/g2_t5/material_graph.json`](../verification/wake_material_graph/g2_t5/material_graph.json)。
它包含主翼和平尾的全部参考坐标、尾迹面片、使用/未使用节点 ID、根、前驱、
排序、每条前驱边的所属面片、最长路径，以及前 20 条长边的上游候选和选择距离。
`material_graph` 字段直接使用 `wake_material_graph_audit` 的原始返回值。
各组节点 ID 是该组 `wake_xyz` 的索引，不能当作合并壁面的全局 ID；
`length_chord` 统一除以记录的 `reference_chord`，路径按尾端到根的方向保存。

输入仍引用本机唯一的 `runs/cases/wingbody/half_cad_wingbody_v1/surface.json`，
SHA-256 为 `d04e439cddb2a80ff9d376257ed9a57d5613f538e111a0ba97f3fbb5f6405d5a`；
平尾仍由原 `build_case(case="g2")` 生成。JSON 自带完整图，可在没有本机 CAD 文件时
直接复核前驱边和累计路径；重新从 CAD 生成则需要上述输入。
代码 SHA、源文件哈希、运行环境和 T5 结果哈希均在 JSON 中。

从仓库根目录、使用项目规定的 myml Python：

```powershell
python -m tools.issue84.t5_gate_ab new_gate_per_node 1 runs/studies/wake_material_graph/g2_t5/t5_gate_ab.json 0.25 --surface runs/cases/wingbody/half_cad_wingbody_v1/surface.json
python -m tools.material_graph_audit --case g2 --surface runs/cases/wingbody/half_cad_wingbody_v1/surface.json --out verification/wake_material_graph/g2_t5/material_graph.json --t5-result runs/studies/wake_material_graph/g2_t5/t5_gate_ab.json
```

第一条是原始 signed-speed Kutta、audit gate、advance 策略和 12 次回溯的 T5 路径，
执行一次更新和更新后的重新求解；回调现在保留每轮 `material_graph`。
第二条只构建和审计冻结的参考图，不做流场求解；可省略 `--t5-result` 单独运行。
指定该参数时，会检查实际 T5 的全部回调记录与独立导出完全一致。

2026-11-05 在 `1548a44` 加本轮脚本改动、Windows x64 / myml Python 3.13.14 实测：

| 组 | 坐标数组节点 | 实际尾迹节点 / 根 | 最大直连边 | 最长累计路径 |
| --- | ---: | ---: | ---: | ---: |
| main_wing | 765 | 468 / 90 | 26.000000c | 30.072260c |
| raised_tail | 20 | 20 / 5 | 0.916667c | 2.750000c |

主翼 51 条 26c 前驱边全部是真实三角网格边，其中 49 条由两个面片共享，
2 条只属于一个面片。比如节点 `374 → 2` 从 `(31, 0.1, 0)` 连到
`(5, 0.1, 0)`，属于尾迹面片 `380, 381`；图中没有环或 x 回折。
因此 26c 确为直接网格连接，不是把累计路径当成直连距离，也不是前驱选择了
非相邻节点。这没有证明长边是可接受的截断边界；不能据此冻结、删除这些边或
宣布塌陷修复。下一步应检查这些面片的网格生成来源与下游截断离散方式。

短 T5 求解的两轮记录均与导出审计完全一致，CL 为 `0.441355395455 → 0.441461695686`，
原始 map residual 为 `22.981516c → 52.015090c`，分别复现历史初始和首步结果。
仅运行一次更新，`status=iteration_limit`，不是八轮 T5 或自由尾迹收敛验收。
针对性 `python -m pytest -q test/test_p1_coupled_material_wake.py test/test_p1_steady_free_wake.py -p no:cacheprovider`
实测 50 passed / 27.10 s（BLAS/OMP/MKL 均 1 线程）；未重跑全套。
尾迹模型文档契约另测 9 passed / 0.14 s；仅用导出 JSON 重建两组审计、逐条复核
前驱边的面片 ID 和长度、校对全部源文件哈希也通过，无需读取本机 CAD 输入。
局部日志与 T5 原始结果在 `runs/studies/wake_material_graph/g2_t5/`，
启动日志在 `runs/scratch/g2_t5_material_graph.log`。算法、几何门槛和默认更新策略未改。

## 10. B3 尾迹位置、离散与根部约束实验

issue #113 的受控实验使用 `solve_steady_free_wake` 的 B3 单翼路径，
不是 §9 的 G2 T5 耦合 material-wake 路径。实验流程使用
ARS-Codex `experiment-agent` 的 run 模式，实际数值证据由本机脚本生成，
不包含跨模型评审或物理精度验收。

输入由 `docs/issue92/b3_cases.json` 的原 CAD 文件和矩形翼参数构建。
冻结全部壁面、初始尾迹、P1 Gamma 基函数、参考量及 pressure-Kutta 设置；
`initial_state.npz` 与逐轮快照记录输入和状态哈希。自由尾迹迭代 0
采用 pressure-Kutta，不能与历史表中的 signed-speed 固定尾迹 CL 混用。
三个 pressure 初值分别为 rect `0.256144450850491`、
swept30 `0.403535637981266`、elliptic `0.278756973432855`。

脚本和单项复现命令如下，均在仓库根目录使用唯一的 myml Python，
设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`：

```powershell
python -m tools.wake_sensitivity_baseline --case swept30 --iterations 60 --out runs/verification/issue102_wake_sensitivity
python -m tools.wake_position_study --phase e1 --case swept30 --out runs/verification/issue102_wake_sensitivity
python -m tools.wake_intervention_study --phase e2 --case swept30 --variant length_5 --iterations 8 --out runs/verification/issue102_wake_sensitivity
python -m tools.wake_intervention_study --phase e3 --case swept30 --variant freeze_1 --iterations 60 --out runs/verification/issue102_wake_sensitivity
python -m tools.wake_position_study --phase e4 --case swept30 --out runs/verification/issue102_wake_sensitivity
python -m tools.wake_sensitivity_audit --case swept30 --out runs/verification/issue102_wake_sensitivity
python -m tools.wake_sensitivity_report --out runs/verification/issue102_wake_sensitivity
```

基线和干预脚本拒绝覆盖冻结输入或已有历史。复算应指定另一个功能输出目录，
或先保留失败尝试；不能覆盖失败来得到“全通过”。多算例的剩余队列由
`tools.wake_sensitivity_queue` 管理，子进程并发上限为 8，单任务超时 7200 s。
`queue.json` 记录实际命令、PID、退出码与超时；`*_error.log` 保留错误。
路径中的 issue102 是 #113 明确要求的证据目录名，属于此实验的兼容例外。

E1 将真实壁面重合的后缘定义为 attachment；网格边 BFS 的第 1 行、
第 1–3 行和第 4 行及以后分别作为 first-free、near-TE 和 downstream。
这些集合不等于按 x 选择前驱得到的 graph roots。每次扰动从同一冻结状态
开始，只改变尾迹坐标；A/J/K/grad 复用，只重装 W。中心差分以
`epsilon/chord` 为自变量，同时报告对称中心偏移，以发现非线性响应。
固定流向探针和初次自由更新方向都按该区域的最大节点位移归一化。

E2 在 `x=max(初始 attachment x)+L*c` 处截断有限 P1 面片，
交点沿原面片线性插值 Gamma hats，公共近尾迹及 Gamma 自由度不变；
截面之后不附加延拓，不调用 `boundary` 策略。流向和展向分别选择投影最长边，
作一次符合共享边的中点细分，P1 基函数精确继承。非结构三角形的斜边同时
带 x/y 分量，因此必须同时读取 dx/dy 分布；这不是纯张量网格的单轴细化证明。
E2 只执行 8 次更新，用于比较初始失稳，不足以声称长时间收敛。

E3 的 freeze_1/2/3 分别固定 BFS 第 0、0–1、0–2 行；
smooth 使用 `smoothstep(row/3)` 位移权重。仍使用原 P1 势、Gamma、
探针、原始目标、轴向保护与 0.3 松弛。每轮分别保存未松弛目标位移、
实际位移、法向滑移、CL/Gamma 和几何检查；这些边界约束仅用于诊断。

E4 捕获原 pressure-Kutta 求解使用的 QR 因子，
测量加权壁面算子在 `null(J)` 上的谱，以及加入压力残差 Jacobian 后的谱。
小于等于 256 列时才使用精确 SVD；其余通过 QR 三角求解配合
Gram/逆 Gram 的 Lanczos 极值估计，同时保存特征对残差及 1 范数条件估计。
不为该实验额外构造大型完整 Jacobian；谱估计不等于精确数值秩证明。

几何审计使用六条真实线段/三角形相交测试、共面正面积重叠与闭合壁面的
立体角 winding。只排除初始声明的公共后缘点/边接触，不放宽接触容差。
最小全局壁面距离为零时，由仍与壁面重合的节点证明；另列非 attachment
节点到壁面三角形的最小距离，不能把后者冒充全局最小距离。
`e1_geometry_audit.json` 独立校正早期实验审计的误接触分类，
原始数值结果及失败日志保留，复核不重新求解或替换 CL/Gamma。

所有输出保持 `physical_accuracy_accepted=false`。原求解器的非线性
success/stationary、jump/Kutta 残差、严格 `wall_over_floor<=1.05`
检查分开报告；后掠翼的冻结 pressure 初值本身不满足最后一项。
几何稳定、滑移收敛、内层收敛和物理精度不能互相替代。
实验执行进度与阻塞只写根目录 `STATUS.md`。

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
