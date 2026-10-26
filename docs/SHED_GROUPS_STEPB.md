# Step B: 右半多脱落组（v2）装配与稠密参考解

本文件记录 `docs/ISSUE39_SHED_GROUPS_V2.md` 契约的 Step B 第一轮实现结果。
Step A（契约）在 PR #40，Step B（代码）在本分支，二者不混在一个 PR。

## 1. 范围

新增（不改动任何已有生产文件）：

| 文件 | 作用 |
| --- | --- |
| `panel/shed_groups.py` | v2 组数据结构、D1–D12 校验、稠密装配、求解与逐组报告 |
| `panel/group_fixture.py` | 研究用夹具构造：单层/两层叠放、抬高尾翼、v2 声明导出 |
| `panel/run_half_groups.py` | 独立研究驱动入口：哈希、真实 exit、块结构/质量行/镜像/全模 oracle/FD-vs-AD |
| `test/test_shed_groups_contract.py` | F0–F8 契约测试（小夹具，9 passed / 9.1 s） |
| `test/test_run_half_groups_entry.py` | 入口契约测试（17 passed / 12.7 s，含真实 CLI 与拒绝路径） |

未改动：`panel/half_native.py`、`panel/kernel.py`、`panel/audit.py`、`panel/cad_half.py`、
`panel/pressure_stencil.py`、`panel/run_half_native.py`。默认路径仍是 v1，
`physical_accuracy_accepted=False` 贯穿所有返回字典。

非目标（明确不做）：侧滑、垂尾/倾斜尾迹、机身涡分离、GPU、训练、
按"CD 更小/更接近 Trefftz"选参数、CLI/OCC 新网格（下一步另开）。

## 2. 装配

一个组 `g` 携带自己的尾迹三角形、自己的一对上下后缘 bank、自己的站位
`y_g` 与自由度 `dof_g`，所以有自己的基函数矩阵 `hat_g`、区间索引、以及

```text
mass_g[i,j]     = sum_{e,q} w_eq B_i(y_eq) B_j(y_eq)
integral_g[e,i] = sum_q     w_eq B_i(y_eq)
measure_g[i]    = sum_e integral_g[e,i]              # 每自由度向量，不是标量
M_g = mass_g / measure_g[:, None]                    # 行和 = 1 仅当基函数单位分解
T_g = (integral_g^T @ incidence_g) / measure_g[:, None]
W_g = (Dd_g + Dd_g^image) @ basis_g + (My_g - My_g^image) @ slope_g
```

系统（`nf` 个壁面、`G` 组、第 g 组 `ng_g` 个自由度）：

```text
[ D        W_1   ...  W_G ] [ mu    ]   [ (Sd+Si) sigma ]
[ -T_1     M_1   ...   0  ] [ g_1   ] = [      0       ]
[  0       ...   -T_G  M_G] [ g_G   ]   [      0       ]
```

`D = Dd + Di` 与 `Sd + Si` 是壁面自身与精确镜像的单层/双层影响（与 v1 同一核、
同一 `interior_diagonal=True`），`W_g` 是组 g 的常值+线性尾迹双层算子。
**仍然不构造全机镜像网格或全机 AIC**：镜像只在核内部瞬态出现，未知量是
`nf + sum(ng_g)`。

## 3. 校验（D1–D12，实际报错串）

| 规则 | 实现判据 | 报错串 |
| --- | --- | --- |
| D1 | 每条 shed edge 是壁面 2-流形上的真实边、且被本组尾迹共享 | `Shed edge must be a real wall edge shared by the group wake` |
| D2 | 该边同时属于声明的上/下 bank 面，且两 bank 不同面 | `Shed edge must be shared by its upper bank face` / `... lower ...` / `Shed upper and lower banks must be distinct faces` |
| D3 | 尾迹常数 z、+z 法向、正半域 | `Only horizontal constant-z wakes are supported in v2` / `Group wake triangles must have +z normals` / `Group wake is not planar at its declared origin_z` |
| D4 | 同一 shed edge 不得跨组重复 | `Duplicate shed edge across groups` |
| D5 | 站位严格递增、`y_0 = 0`、`dof[-1] = -1`、自由度连续且都出现 | `Group circulation must vanish at the outboard tip` / `Group circulation DOFs must be contiguous and present` |
| D6 | 每个尾迹三角形与每个 TE 求积点都落在固定站位条带内 | `Group wake/TE left the fixed circulation strips` |
| D9 | `symmetry_plane` 根贴 y=0；`buried_in_solid` 根不贴 y=0 且在声明组件的包围盒内 | `Symmetry-plane root must touch the y=0 plane` / `Buried root must not lie on the symmetry plane` / `Buried root is not inside the declared component` |
| D10 | 尾迹不得切开其它壁面（AABB 预筛 + 边-三角形 Möller） | `Group wake intersects a wall panel` |
| D11 | 合法性只来自拓扑 + 可选 `role`/`component_roles` 交叉校验，**不读 CAD 组件号** | `Shed group role contradicts the declared component roles` |
| D12 | 尾迹是圆盘：内部边恰 2 面；边界边只允许在 TE、y=0 根线、翼尖站位线、下游截断处 | `Group wake has a non-manifold interior edge` / `Group wake boundary is not a declared TE, root line, tip line or downstream truncation` |

D12 的实际边界规则比契约初稿更精确：有限尾迹是一个**带边界的薄片**，不是闭合
2-流形，旧写法（"必须闭合"）在本夹具上会把合法尾迹判成非法。这一点已在
实现中修正，并在下文"与契约的差异"里列出。

D11 的反面例子就是 `runs/geom_robust/cad/bwb`：机体一体、没有独立机翼组件号，
但存在真实锐后缘。因此判据必须是拓扑，不是组件号。

## 4. 契约测试 F0–F8

夹具：`cases/panel_half/thick_wing_two_strip.json`（20 壁面 / 4 尾迹 / 2 自由度，
闭合厚翼）在 z 方向复制一份（`dz = 0.4`），得到两个几何独立的组：
`wing_low`（z=0，贴对称面）与 `wing_high`（z=0.4）。壁面 40 面、尾迹 8 面、
未知量 44。

| 用例 | 内容 | 结果 |
| --- | --- | --- |
| F0 | G=0（纯闭合机身）：只解 `D mu = S sigma` | 未知量 = 40，残差/右端 = 2.6e-16 |
| F1 | G=1 必须与 v1 `solve_half_fields` 逐位一致 | `mu` 0.0，`gamma` 0.0，`Cp_raw` 0.0，`CL` 差 0.0，`CD_p` 0.0 |
| F2 | 已知强迫子问题：`b - W_2 g_2` 解 wall+g_1，必须复现完整解 | < 1e-12（相对） |
| F3 | 真删组后其余解**允许**变化，差值 = `A_1^{-1}[W_2 g_2; 0]` | < 1e-12；且 `removed != full`（>1e-6），否定"删组不变"的错误断言 |
| F4 | 单独一组建在 z=0.4 时也与 v1 一致（组级 z 不是硬编码 0） | `mu` / `gamma` < 1e-13，`CL` 差 < 1e-12 |
| F5 | 六类必拒输入 | 全部按上面的英文串拒绝 |
| F6 | 两组真实解：残差、逐组 `gamma`/`te_jump`/`kutta` 行 | 残差/右端 2.5e-16，Kutta 行残差 0.0，两组 `gamma` 分别 0.3297 / 0.0586 |
| F7 | 只翻一组 Kutta 行符号 → **两组都变**（只能经 mu 耦合） | 组 1 变化 >1e-3，组 2 >1e-2，残差仍 < 1e-12 |
| F8 | 伴随配对 `y·b = c·x`（稠密转置） | ≤ 1e-10（相对） |

两组真实解读数（5° 迎角、`Sref=2`、`Cref=1`）：`CL=0.124974825724`、
`CD_p=0.022288087201`、`CM_y=-0.092604515313`；`wing_low` 的 TE 势跳跃 RMS
0.145074，`wing_high` 0.039695。这个夹具是**结构测试**，不是物理预测：
20 个面元、两层叠放且间距 0.4 弦长，互干扰很强，不能据此谈气动精度。

## 5. 与契约初稿的差异（需要同步到 PR #40 的文档）

1. D12 由"闭合 2-流形"改为"圆盘 + 四类合法边界"，原因见第 3 节。
2. 契约拒绝表里的 `Attached lifting shed group requires a verified physical TE and bank topology`
   在本轮实现中由更具体的三条 D1/D2 错误串承担（真实边 / bank 归属 / 尾迹共享），
   语义等价但文本不同；若需要逐字对齐，应改契约表而不应放宽判据。
3. `buried_in_solid` 的根部容纳判据目前是"包围盒 + 不在该组件壁面上"，
   实现里如实标为 `root_containment = aabb_and_edge_distance`，不是严格的内外测试。

## 6. 诚实的限制

1. **D10 只做边-三角形相交**（AABB 预筛 + Möller），能抓住真实穿插，
   但不是三角形-三角形 SAT 完整证明；报告里不声称"无交"。
2. **尾迹必须水平（常数 z）**。抬高平尾的 T 尾结构在拓扑上可以有两组尾迹，
   但倾斜尾迹尚未实现，这也是后续改 `cad_half.py` 的前置条件。
3. **没有场级双侧指标**（`Vmean·n`、`Vmean·grad(Gamma)`、两侧 `DeltaCp`）。
   那套独立审计在 issue #28 / PR #30 的 `panel/audit_half_wake.py` 里，
   本模块只报逐组 `gamma`、TE 势跳跃、Kutta 行残差、尾迹常值/线性输运量，
   不重复实现第二套场评估。
4. **没有 CLI、没有 OCC 生成、没有微分**。数学与装配先验收，可微版本要单独过
   一遍，不能假定稠密参考路径的正确性自动延伸到 VJP。
5. 仍然是稠密 LU：`O((nf + sum ng)^2)` 内存。规模上限由使用方门控，
   本模块不含 `--max-panels` 之类的策略。

## 7. 复现

```powershell
$env:TMP=$env:TEMP=$env:TMPDIR='D:\3D\Projects\panel\runs\pipwork'
Set-Location 'D:\3D\Projects\panel\runs\pr41_git'
& 'D:\3D\Projects\panel\.conda_env\python.exe' -m pytest -q test/test_shed_groups_contract.py -p no:cacheprovider
```

环境：Python 3.11.17、JAX 0.10.2 CPU（测试内 `jax_enable_x64=True`）、NumPy 2.4.6。

## 8. 研究驱动入口 `panel/run_half_groups.py`

issue #41 要求的不只是数学，还要一个可复现的实验入口。它与 `run_half_native` 并列，
**不改任何默认**：v1 仍是默认路径，入口只在研究 PR 里被显式调用。

```powershell
$env:TMP=$env:TEMP=$env:TMPDIR='D:\3D\Projects\panel\runs\pipwork'
Set-Location 'D:\3D\Projects\panel\runs\pr41_git'
$py = 'D:\3D\Projects\panel\.conda_env\python.exe'
& $py runs\peak_wrap.py panel.run_half_groups runs\pr41_verify\peak_selfcheck --self-check --out runs\pr41_verify\selfcheck.json
& $py -m panel.run_half_groups --surface <surface.json> --groups <groups.json> --out <report.json>
```

报告 schema `dpanel.half_group_report.v1`，字段：

| 字段 | 内容 |
| --- | --- |
| `inputs` | `surface`/`groups` 路径与 SHA-256、声明 schema、源 surface 的 `audit`/`provenance` |
| `code` | Python/NumPy/JAX/平台 + `panel/{shed_groups,run_half_groups,half_native,kernel}.py` 的 SHA-256 |
| `groups` / `validation` | 每组 kind/role/根条件/尾迹面数/shed 边数/自由度/尾迹 z；D 规则审计 |
| `solve` | 未知量、壁面数、组切片、真残差（`||r||/||b||`）、热解秒数、CL/CD_p/CM/Cp 极值 |
| `group_rows` | 逐组 `gamma`、`gamma_sum`、TE 势跳跃 RMS/max、Kutta 行残差、尾迹常值输运 |
| `checks` | `block_dimensions`、`circulation_mass_rows`、`kutta_rows`、`operator_mirror`、`v1_equivalence`、`full_oracle`、`lift_slope` |

退出码：成功 0；被拒绝的输入 2（traceback 与 `status="refused"` 写进报告文件）；意外崩溃 1。
`--out` 已存在时直接 `FileExistsError`，不静默覆盖。

### 8.1 `--self-check` 的四个算例（本机实测，全部 exit 0）

| 算例 | 壁面/尾迹/未知量 | CL | CD_p | CM_y | 真残差 |
| --- | --- | --- | --- | --- | --- |
| G0（无尾迹） | 20 / 0 / 20 | 2.5e-16 | 6.2e-17 | −0.023112 | 2.03e-16 |
| G1（z=0 单组） | 20 / 4 / 22 | 0.083285 | 0.009492 | −0.044459 | 1.98e-16 |
| G2（叠放 dz=0.4） | 40 / 8 / 44 | 0.124975 | 0.022288 | −0.092605 | 2.51e-16 |
| G2_tail_z024（抬高尾翼 dz=0.24） | 40 / 8 / 44 | 0.307546 | 0.133417 | −0.279616 | 5.97e-16 |

`G2_tail_z024` 的第二组 `gamma_sum = -0.5812`（第一组 +1.1361）：把"上表面"沉到下方
组的尾迹里，环量可以整体反号。这也再次证明**叠放对不是对称对**，不能假设两组同值。

### 8.2 块结构、质量行与镜像

| 检查 | G1 | G2 |
| --- | --- | --- |
| 矩阵形状 / 未知量 | 22 × 22 | 44 × 44 |
| 组切片 | `[[20,22]]` | `[[40,42],[42,44]]` |
| Kutta 行残差 RMS | 9.8e-18 | 0.0 |
| `M_g = raw_mass/measure` 归一化差 | 0.0 | 0.0 |
| TE 求积重建差（独立 einsum） | 5.6e-17 | 5.6e-17 |
| 壁面双层块 vs 公式重建（非对角） | 6.9e-18 | 2.8e-17 |
| 尾迹算子 vs 显式重建 | 0.0 | 0.0 |

**关于"M 行和恒为 1"的精确化**：行的和等于 1 需要该行所有 TE 求积点上基函数满足单位分解。
夹具的翼尖站位是 `dof = -1`（环量在翼尖端处收为零），因此最外一段的基函数之和是
`1 - frac < 1`，实测 `row_sums = [1.0, 0.8333]`、`basis_partition_max_abs_error = 0.7887`。
报告因此同时给出 `row_sums` 与 `row_sums_where_partition_of_unity`（后者为 `[1.0]`），
**不把该行强行归一**。契约表格里的"行和为 1"需加这个前提。

### 8.3 全模 oracle 与可微性

- **oracle（G1）**：走已验证的 `make_full_mirror_topology` + `solve_fields`，40 个全模壁面，
  `mu` 相对峰值差 1.60e-15、`gamma` 2.57e-15。全模的环量未知量是 3 个（半模 2 个 +
  一个镜像副本），比较时必须取前 `ng` 个；否则报 `operands could not be broadcast (3,) (2,)`。
  多组不提供 oracle：目前没有多尾迹的全机模型，如实返回 `performed=False`。
- **梯度**：`dCL/dalpha` 用前向 AD 与中心差分（`h = 1e-4, 1e-5`）对照。G1 0.842831 /rad
  （0.0147102 /deg），相对差 1.09e-08 / 1.19e-10；G2 0.806895 /rad（0.0140830 /deg），
  1.09e-08 / 1.06e-10；G2_tail 0.470933 /rad（0.00821933 /deg），1.09e-08 / 6.78e-11。
  G0 没有环量未知量，CL 恒为 0，梯度检查如实跳过。

### 8.4 与契约初稿的第二批差异

1. **"M 行和为 1" 需加"基函数单位分解"前提**（见 8.2）。
2. **全模环量未知量编号**：全模在末尾追加镜像副本，只在 `[0, ng)` 上与半模可比；
   契约的镜像规则 M1–M6 应写明这条切片约定。
3. **entry 的 `--self-check` 是独立路径**：G0（无尾迹）与 G1（单组）都从同一夹具派生，
   但 G0 的 surface 没有 `wake_faces` 字段内容，入口因此**显式**构造空组列表，
   而不是伪造一个空尾迹去骗过校验。

### 8.5 资源与诚实限制

- `--self-check` 四例合计峰值 RSS **1172.4 MB**、墙钟约 2 分钟（四例各自 JIT 编译 +
  每例一次 AD 与两次差分）。相对 44 个未知量这很高，成本在编译与核评估，不在 LU。
- 逐例 JIT 缓存没有复用（每个算例都重编译），这是后续可优化点，不影响数值结论。
- D10 仍只做边-三角形相交；尾迹仍是常数 z；无场级双侧指标；稠密 LU。

## 9. 下一步

1. 把第 5 节与 8.4 的差异同步回契约（PR #40）。
2. 真实 CAD 两片独立尾迹：**已完成**，见 `docs/SHED_GROUPS_CAD.md`
   （新增 `panel/wake_group_declaration.py`、`panel/cad_half_groups.py`、
   `panel/half_group_surface.py` 与两片 z = 0 / z = 0.24 尾迹的真实 STEP 夹具；
   全套 37 passed；两档实测 CL 0.285245 -> 0.298993，近场 CD_p 仍为负）。
3. 把两片尾迹的声明与 manifest 字段补进契约附录（PR #40）。
4. 场级双侧指标复用 PR #30 的审计模块，不新建第二套。
