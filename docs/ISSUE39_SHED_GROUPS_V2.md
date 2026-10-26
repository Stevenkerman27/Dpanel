# issue #39 Step A：shed_groups v2 契约（文档，不含求解器实现）

状态：契约草案，供审查。**本文件不改变任何生产数值路径**，不新增求解器代码，
不改变默认模式；`physical_accuracy_accepted=false` 不变。
关联：#37（整机几何/载荷需求）、#20/#21/#22（半模网格与原生算子）、#36（制造势验证）、
PR #30（物理尾迹审计）。Step B 的实现按本契约另开 PR。

## 0. 本契约要解决什么

v1（`dpanel.half_planar_cut.v1`）只有一个后缘多值函数 `x(y)`、
一张尾迹、一组 `gamma_stations/gamma_station_dof`，并且尾迹三角形必须
**全部**在同一 z 平面（`panel/half_native.py:154-155`
`Only a constant-z planar wake is supported`），网格侧的尾迹点更是硬编码
`z=0`（`panel/cad_half.py:178`）。于是常规布局里抬高的平尾只能当
Dirichlet 湿壁面，拿不到独立环量、尾迹与 Kutta 条件
（`cases/geom_robust/conv/build.json` 已记录这一点）。

v2 的目标不是「给每个零件都挂尾迹」，而是**按物理脱落线声明**：默认光滑机身
`shed_groups=[]`，升力面各占一组，允许每组有自己的后缘曲线、自己的尾迹高度、
自己的 `station_y` 与 `gamma_station_dof`。

## 1. 顶层结构

```text
dpanel.half_shed_groups.v2
├─ schema, parent_schema, units, provenance
├─ geometry      : coordinates, moment_reference, reference_area/chord, density, freestream
├─ wall          : faces, component_ids            (单一 mu/sigma 场共享)
├─ contact_cuts  : 全局切边（压力模板用，不是 Kutta 行）
├─ shed_groups[] : 0..G 组，见第 2 节
└─ compat        : v1 映射规则（第 6 节）
```

| 字段 | 类型 | 必需 | 语义 |
| --- | --- | --- | --- |
| `schema` | string | 是 | 固定 `dpanel.half_shed_groups.v2` |
| `parent_schema` | string | 是 | 祖先格式，v1 迁移时写 `dpanel.half_planar_cut.v1` |
| `units` | object | 是 | `length`/`velocity`/`density`，全部 SI；见第 9 节 |
| `geometry.coordinates` | (n,3) float | 是 | 右半网格节点，`y>=0` |
| `geometry.moment_reference` | (3,) float | 是 | 必须在 y=0 对称面上 |
| `geometry.freestream` | (3,) float | 是 | `v_y==0`（零侧滑）；v2 不表示侧滑 |
| `wall.faces` | (m,3) int | 是 | 物理湿壁，含机身与所有升力面；右半 |
| `wall.component_ids` | (m,) int | 是 | CAD 组件号，仅诊断/分部件载荷；**不是**尾迹许可（D11 不读它） |
| `component_roles` | object | 否 | `{组件号: lifting|non_lifting|unclassified}`，只用于与组 `role` 交叉校验和报告；缺省时按拓扑判定 |
| `contact_cuts` | (c,2) int | 否 | 真实壁面边，切边不参与压力模板邻接 |
| `shed_groups` | list | 是 | 长度 0..G，见第 2 节 |

**G=0 是合法配置**，表示纯闭合机身（无尾迹、无 Gamma）。v1 无法表达它
（`panel/half_native.py:160-164` 要求 `ng>=1` 且 DOF 连续）。
G=0 时偶极/质量块整块消失，只剩 `D mu = S sigma`；第 8 节 F0 给它的最小验收。

## 2. shed_groups 的字段

每组是**一段脱落线**，不是一个 CAD 零件：

| 字段 | 类型 | 必需 | 语义 |
| --- | --- | --- | --- |
| `group_id` | string | 是 | 唯一名，报告与诊断用（如 `wing`、`htail`） |
| `kind` | enum | 是 | `attached_lifting_TE` 或 `prescribed_sharp_separation` |
| `source_component_ids` | int[] | 是 | 只作出处记录；**不自动授权尾迹** |
| `role` | enum | 否 | `lifting` / `non_lifting`；缺省时按拓扑判定（D1/D2/D9），**不读组件号**；与顶层 `component_roles` 冲突即拒绝（D11） |
| `shed_edges` | (e,2) int[] | 是 | 真实网格边，必须同时属于本组 upper/lower bank 各一张面，且被本组尾迹的某条边共享 |
| `upper_bank_faces` / `lower_bank_faces` | int[] | 是 | 后缘两侧面片索引，长度与 `shed_edges` 相同 |
| `wake_faces` | (w,3) int | 是 | 本组尾迹三角形（全局节点索引） |
| `wake_material_labels` | string[] | 否 | 每片尾迹的材料/来源标签，镜像与诊断用 |
| `gamma_stations` | (s,) float | 是 | 本组沿 y 的站位，**严格递增**，`s>=2` |
| `gamma_station_dof` | (s,) int | 是 | `-1` 表示该站位无自由度（翼尖/端板/埋根），其余取连续 `0..ng-1` |
| `closure` | object | 是 | `kutta: weak_mass or none`、`basis`、`applicability`、`source` |
| `root_condition` | enum | 是 | `symmetry_plane`（翼根在 y=0 是真实对称面）或 `buried_in_solid`（根部埋进另一实体，T 尾/背鳍） |
| `wake_plane` | object | 是 | `{normal:[0,0,1], origin_z: value}`；v2 只支持水平（常 z）尾迹，见第 11 节 |
| `contact_with` | object | 否 | 本组尾迹与其它组/机身共享的真实切边及归属规则 |

G 组的 Gamma 全局编号为 `offset_g + 0..ng_g-1`，
`offset_g = sum(ng_h for h<g)`；总未知量 `N + sum(ng_g)`。

## 3. 拓扑恒等式（逐条可局部检查，违反即拒绝）

记号：`E_g` 为 `shed_edges`，`W_g` 为尾迹面集合，
`B_g_up / B_g_lo` 为两侧 bank 集合。

- **D1 真实边**：每条 `e in E_g` 是墙面的真实边，且是 `W_g` 中某张
  尾迹三角形的边。v1 对应检查 `panel/half_native.py:142-152`。
- **D2 bank 归属**：`e` 的两个顶点都出现在 upper 与 lower bank 面里，两张面不同。
- **D3 bank 不相交**：同组 up/lo 面集合不相交，不同 `e` 的 up/lo 面不重复。
- **D4 尾迹只挂本组**：`W_g` 与其它组没有共享三角形；同一 `e` 不能出现在两组。
- **D5 位面**：`W_g` 所有顶点满足 `|z - origin_z| <= tol`；v2 不做倾斜/垂直尾迹。
- **D6 正侧**：所有墙/尾迹形心 `y>0`；节点 `y>=0`；不许出现 y=0 上的整片三角形
  （对称 cap）。v1 对应 `panel/half_surface_mesh.py:65` 与
  `panel/half_native.py:137-139`。
- **D7 DOF 连续**：每组 DOF 取值集合恰为 `0..ng_g-1` 加 `-1`，且
  `ng_g>=1`；无尾迹用 `shed_groups=[]` 表达，不许用「空组」表达。
  v1 对应 `panel/half_native.py:160-164`。
- **D8 站位条带**：本组尾迹形心 y 落在本站位区间内；后缘边两端 y 落在同一站位条带内。
  v1 对应 `panel/half_native.py:174-179`。
- **D9 根条件**：`symmetry_plane` 时翼根边必须正好在 y=0 上；
  `buried_in_solid` 时翼根边**不得**在 y=0 上，且整体位于另一实体墙面内部
  （T 尾根部埋进垂尾）。这一条直接对应 conv 案例。
- **D10 不穿体**：`W_g` 与任何墙面（除本组 shed 边）不得三角形相交；
  先 AABB 粗筛再显式三角-三角相交测试，只报告不自动修复。
- **D11 Kutta 合法性由物理 TE 决定，不由组件号决定**：`kind=attached_lifting_TE`
  的组必须 (a) 每条 `shed_edges` 都是上下两 bank 共享的真实网格边且被本组尾迹
  共享（即 D1/D2 已过）、(b) `closure.applicability` 显式声明适用域、(c) 若输入
  带 `component_roles` 元数据，则组的 `role` 必须与之相符。
  **不得**用「组件号只含机身」直接拒绝：BWB 的机身组件同样可能有真实锐后缘并
  合法脱落（`runs/geom_robust/cad/bwb` 就是机体一体、没有独立机翼组件）。
  要禁止某个实体脱落，必须由 `component_roles`（`lifting` / `non_lifting` /
  `unclassified`）显式声明，或由组自身 `role=non_lifting` 声明，不用 CAD 组件号
  20/30 做判据。
- **D12 尾迹无自交**：`W_g` 的节点-边邻接是 2-流形，无重复三角形、无孤立节点。

## 4. 线性块方程

设墙偶极强度 `mu in R^N`，各组环量 `gamma_g in R^{ng_g}`，
`D = Dd + Di`（直接 + 镜像偶极核），`S = Sd + Si`（直接 + 镜像源核），
`sigma = -(n . U_inf)`。第 g 组在控制点上：

```text
W_g = (dw_g + dw_g_img) @ basis_g + (my_g - my_g_img) @ slope_g
T_g = (integral_g^T @ incidence_g) / measure_g        # 弱 Kutta 行
M_g = mass_g / measure_g[:, None]                      # 环量质量行
```

**`M_g` 的精确形式**（照抄 v1 `panel/half_native.py:308-317`，不是近似）：

```text
mass_g[i,j] = sum_{e,q} w_eq * B_i(y_eq) * B_j(y_eq)      # (ng, ng)
integral_g[e,i] = sum_q w_eq * B_i(y_eq)                  # (E, ng)
measure_g[i] = sum_e integral_g[e,i] = sum_{e,q} w_eq B_i(y_eq)   # (ng,) 每自由度
T_g          = (integral_g^T @ incidence_g) / measure_g[:, None]
```

`measure_g` 是**每自由度的基函数积分向量**（`integrals.sum(axis=0)`），不是标量。
写成 `(integral_g^T @ integral_g) / measure_g` 是错的：积分权重必须在两个基函数
**乘积之后**相乘，`w_eq B_i` 与 `w_eq B_j` 单独求和再相乘会得到
`w_eq^2 B_i B_j` 类项；非均匀真实后缘上两者差别可达 78%（附录 A），`ng=1` 时
直接差一个 `measure` 因子（量纲错误），并会破坏第 8 节 F1 的 `G=1` 逐位基线。
可检验的恒等式：basis 是单位分解（`sum_i B_i = 1`）时 `M_g` 的**行和恒为 1**，
错误公式不满足。

其中 `basis_g`、`slope_g` 由本组 `gamma_stations` 与尾迹
形心 y 线性插值；`incidence_g` 在 upper bank 上 +1、lower bank 上 -1。
镜像半的 `slope` **反号**，因为反射使 y 导数变号
（`panel/half_native.py:341-342`）。整体系统：

```text
[ D          W_1   W_2   ...  W_G ] [ mu      ]   [ S sigma ]
[ -T_1       M_1   0     ...  0   ] [ gamma_1 ]   [ 0       ]
[ -T_2       0     M_2   ...  0   ] [ gamma_2 ] = [ 0       ]
[  ...       ...   ...   ...  ... ] [  ...    ]   [ ...     ]
[ -T_G       0     0     ...  M_G ] [ gamma_G ]   [ 0       ]
```

三点必须写清：

1. 各组之间**只通过 mu 耦合**；`gamma_g` 之间没有直接块（右下角块对角）。
   可检验含义：同时删掉 `-T_h` 行与 `W_h` 列后，剩下的
   `(mu, gamma_g)` 子问题与整系统在该子空间上的解一致到舍入。
2. `G=1` 必须与 v1 逐位相同（`A = [[D, W],[-T, M]]`、
   `rhs=[S sigma; 0]`，见 `panel/half_native.py:343-345`）。
3. `G=0` 时 `A = D`、`rhs = S sigma`，只有 mu。理想无旋闭合体
   净力应为零，但离散面板解**不保证**逐位为零；任何非零分量必须报告为离散量，
   不能当物理升力。

## 5. 镜像与对称规则

- **M1 映射**：右半节点 `x` 映射到 `x*[1,-1,1]`；墙面三角形顶点序反转以保
  外法向；尾迹同理；稠密 `E` 不构造（`panel/half_native.py:40-42`、
  `panel/verify_half_mirror.py:32-124`）。
- **M2 环量**：每组 `gamma_g` 在镜像半复制，`d gamma/dy` 反号，因此
  `W_g` 的 `slope` 项在镜像半取负；组与组之间互不影响。
- **M3 站位共享**：y=0 上的节点两半共用一个 ID，镜像后不得出现重复节点。
- **M4 埋根例外**：`root_condition=buried_in_solid` 的组，其翼根**不是**对称面，
  镜像不会在那里产生新物理面，只复制积分贡献；该组 `gamma_station_dof[0]`
  必须为 `-1`，并在报告里显式记 `root_condition`。
- **M5 完全镜像 oracle**：v2 验收要求全模两组参考解，但该参考解本身要求 Step B 的
  全模实现；Step A 阶段只验收**映射层**：partner 距离、法向配对、slope 反号、
  mu 对称、gamma 复制，全部到 1e-14。
- **M6 不承诺的对称**：侧滑非零时垂尾气动不对称，半模+镜像给不出；v2 沿用
  `validate_half_configuration` 的零侧滑拒绝（`panel/half_native.py:207-210`）。

## 6. v1 兼容与迁移

- v1 文件按**单组**读取：`group_id="legacy"`、`kind="attached_lifting_TE"`、
  `shed_edges=trailing_edges`、`root_condition="symmetry_plane"`、
  `wake_plane.origin_z` 取尾迹节点唯一的 z。
- 迁移器只做字段重排与断言（D1-D12 全过才写 v2），**不重网格、不重解**。
- v2 读取器必须拒绝 schema 不匹配的输入（沿用 `panel/run_half_native.py:32-33`
  的风格），且**不得**在缺字段时静默把该组退化成无升力壁面。

## 7. 拒绝规则表

| 条件 | 报错（英文原文，便于测试断言） | 来源 |
| --- | --- | --- |
| schema 不匹配 | `Expected dpanel.half_shed_groups.v2` | 新 |
| kind 非法/缺失 | `Shed group kind must be attached_lifting_TE or prescribed_sharp_separation` | 新 |
| 组无自由度 | `Lifting shed group requires at least one circulation DOF` | 新 |
| DOF 不连续 | `Group circulation DOFs must be contiguous and present` | v1 :160-164 |
| 边不是真实边/未挂尾迹 | `Shed edge must be a real wall edge shared by the group wake` | v1 :142-152 |
| bank 相同或相交 | `Shed upper and lower banks must be distinct faces` | v1 :151 + 新 |
| 尾迹离开站位条带 | `Group wake/TE left the fixed circulation strips` | v1 :174-179 |
| 尾迹不在声明 z | `Group wake is not planar at its declared origin_z` | v1 :154-155 |
| 倾斜/垂直尾迹 | `Only horizontal constant-z wakes are supported in v2` | 新 |
| 对称 cap 进壁面 | `Artificial symmetry cap must not be included in physical wall` | mesh :65 |
| 埋根却给了 y=0 根边 | `Buried root must not lie on the symmetry plane` | 新（D9） |
| 组缺物理 TE/bank 拓扑 | `Attached lifting shed group requires a verified physical TE and bank topology` | 新（D11） |
| `role` 与组件角色冲突 | `Shed group role contradicts the declared component roles` | 新（D11） |
| 尾迹穿体 | `Group wake intersects a wall panel` | 新（D10） |
| 相同边属于两组 | `Duplicate shed edge across groups` | 新（D4） |
| 非零侧滑 | `Half model requires y>=0, zero sideslip and reference on symmetry plane` | v1 :207-210 |

失败必须**整体拒绝并保留旧报告**，不允许部分组生效。

## 8. 小夹具验收协议（Step A 只定义，Step B 执行）

固定夹具 `cases/panel_half/shed_groups_two_height.json`（Step B 新建）：
两块平板翼，半展 1.0、弦 1.0、NACA 0012；翼 1 在 z=0，翼 2 抬到 z=0.20，
两后缘各有独立水平尾迹且互不相交；两组各 6 站位、5 DOF，`G=2`。

| 编号 | 验收点 | 判据 |
| --- | --- | --- |
| F0 | `G=0` 纯闭合机身（半球+圆柱）可读取可解 | 真残差 `<1e-12`；报告 `circulation_unknowns=0`；净力只作诊断输出 |
| F1 | `G=1` 与 v1 逐位相同 | mu/Gamma/Cp/力/矩 相对差 `<1e-14` |
| F2 | 耦合只经 `mu`（已知强迫子问题） | 取完整解 `(mu,gamma_1,gamma_2)`，以 `b - W_2 gamma_2` 为第一组右端解 `[D,W_1;-T_1,M_1][mu;gamma_1]`，必须复现完整解，`<1e-12` 相对。另报「只改翼 2 入射」时 `gamma_2` 变化 `>1e-3`，`gamma_1` **允许变化**（经 `mu` 耦合），不设「gamma_1 不变」判据 |
| F3 | 真删组的变化必须可预测 | 真正移除组 2（删 `-T_2` 行与 `W_2` 列）后 `(mu,gamma_1)` **允许**与完整解不同；差值必须等于 `A_1^{-1}[W_2 gamma_2; 0]`，`<1e-12` 相对。**「删组后其余未知量与完整解一致」在数学上不成立**（`D mu + W_1 gamma_1 + W_2 gamma_2 = b` 仍经 `mu` 耦合），不得作判据 |
| F4 | 镜像映射层 | partner 距离、法向配对、slope 反号、mu 对称、gamma 复制，全部 `<1e-14` |
| F5 | 秩与拒绝 | 重复 shed 边、两组共享边、埋根却挂 y=0 根边、`role` 与 `component_roles` 冲突四种输入必须分别抛出第 7 节对应英文错误 |
| F6 | 两组真实残差 | `norm(Ax-b)/norm(b) < 1e-12`，逐组报后缘势跳跃与 `Delta Cp` |
| F7 | 尾迹双侧物理量 | 每组报 `Vmean.n/U`、`Vmean.grad(Gamma)/U^2`、两侧 `Delta Cp`，按固定 `x_gap/Cref` 分段（沿用 PR #30 口径） |
| F8 | 隐式伴随 | `seed . x = adj . b` 到 `1e-12` 相对；两组各自可单独取 VJP |

**明确不声称**：F0-F8 全绿只证明装配、独立性与拒绝规则正确，**不**证明平尾升力、
配平力矩或整机 CD 达到物理精度。任何物理接受都需要同几何的独立 VSPAERO/CFD 对照，
且必须单列。

## 9. 单位与参考量

- `Gamma(C) = integral_C V . dl` 只在**指定流体闭路 C** 后有物理意义；本契约把 C
  定义为「穿过本条 shed edge、沿本组尾迹与壁面闭合的回路」，并要求在测试里用偶极子
  单位检查锁定数值归一化（不在此处硬写公式）。
- 每组 Gamma 的参考量写进报告：`U_inf * c_ref`（量纲上等价于环量），
  但**不得**把 `Gamma_body = L_body/(rho U b)` 当作独立物理状态。
- 分部件载荷优先用真实湿面 `-q Cp n A` 求和；`component_ids`
  未经 CAD 名称证实前只按 ID 报。

## 10. Step B 实施计划（不在本 PR）

范围：只实现**双水平升力面**（主翼 + 抬高平尾）固定拓扑，保留 v1 基线。

1. `panel/shed_groups.py`：v2 读取/校验（D1-D12）、v1 迁移器、拒绝表。
2. `panel/half_native.py`：新增 `solve_half_groups`，按第 4 节拼 `A`；
   v1 路径不动，`G=1` 必须逐位复现。
3. 测试：`test/test_shed_groups_contract.py`（F0-F8）、
   `test/test_shed_groups_v1_compat.py`；**先在小夹具上完成装配与拒绝规则全绿**，
   再碰网格。
4. 之后才改 `panel/cad_half.py`：允许按组声明尾迹平面高度（去掉硬编码 z=0），
   仍只支持水平面，网格仍用 OCC `fragment`。
5. 真实几何：固定 conv STEP 重建（新目录），**比较的是两个不同模型**，
   不能逐面强制相等；报告两组 Gamma、平尾局部 Cp/力矩、总壁面无穿透、
   每组尾迹 `Vmean.n` 与 `Delta Cp`。
6. 退出条件：F0-F8 全绿；v1 12 个既有集成测试零回归；
   `physical_accuracy_accepted=false`；不合并、不训练、不改 GPU 路径。
7. **不与网格工作混做**：壁面/前缘局部加密与网格收敛是独立 issue/PR；
   本轮 A/B 只改「尾迹分组与 Gamma 分组」这一个因素，几何/网格冻结。

## 11. 非目标与风险

- **不做分离涡**：大迎角圆机身的涡升力/涡破裂需要显式分离线、涡丝或 RANS 参考；
  开放 `shed_groups` 接口不等于已经能预测涡升力、基底阻力或压力 Kutta。
- **不做倾斜/垂直尾迹**：垂尾尾迹是 y 向法向的竖直面，需要 v3；v2 显式拒绝。
- **不做侧滑**：半模+镜像给不出非对称垂尾气动。
- **不做接触拓扑变化**：埋根接触边属固定拓扑，几何设计变量不得移动它。
- **风险 1**：`buried_in_solid` 的根部在离散上仍是 y=0 截断；若 T 尾根弦没有
  完全埋进垂尾，会暴露一个人工端面；D9/D10 只能检测不能修复。
- **风险 2**：每组质量矩阵条件数随细化上升（v1 实测 16.9-21.5），两组叠加可能更差；
  Step B 必须逐组报条件数。

## 12. 与整机需求的关系

GRASS-plane 需要整机几何/载荷，但本契约只覆盖「多升力面的物理表示」。
整机配平、总阻力与训练 reward 仍需外部验证与单独的物理接受流程；本条与
#36/#30 口径一致：先证明装配与诊断正确，再谈物理精度。

字段形状的可机读示例见 `docs/issue39_schema_example_v2.json`（仅示例，
当前没有任何 loader 读取它）。

## 附录 A：`M_g` 错误公式的反例（非均匀真实后缘）

纯数学反例，不含求解器。取一组后缘边 `y = [0.05, 0.15, 0.35, 0.70, 0.95]`、
长度 `L = [0.20, 0.05, 0.30, 0.10, 0.25]`（刻意非均匀，模拟真实后缘折线），
每边 2 点 Gauss（`w_eq = L_e/2`，`B_0 = 1-y`，`B_1 = y`）：

| 量 | 数值 |
| --- | --- |
| `measure_g = integrals.sum(axis=0)` | `[0.47, 0.43]` |
| `mass_g` | `[[0.353, 0.117], [0.117, 0.313]]` |
| `M_g = mass_g / measure_g[:, None]`（正确） | `[[0.751064, 0.248936], [0.272093, 0.727907]]` |
| `(integral^T integral) / measure_g`（草案错误式） | `[[0.163803, 0.059069], [0.064564, 0.168576]]` |
| 最大绝对差 | **0.587261**（占 `max|M|` 的 78.2%） |

自检：正确 `M_g` 的每行和恰为 `1`（`B_0+B_1 = 1` 的单位分解），错误式行和
`0.2229/0.2331` 不为 1。

`ng=1`（单常值基 `B_0=1`，`measure=0.8`）：正确 `M=1.0`，错误式 `0.425` —— 差一个
`measure` 因子，纯量纲错误。

这个反例由 `test/test_shed_group_mass_identity.py` 固化为回归测试（纯 NumPy，
不 import JAX，0.1 s）。

