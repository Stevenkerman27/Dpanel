# 两个大 issue 留下了什么：#77 与 #92

这两条线在文档里出现得最多，也最容易让人误判“现在到底还生效不生效”。结论：
**两条的主体都已完成并进入现役代码，剩下的都是明确的未验收项。** 细节长文在
`docs/archive/issue/`，本文只保留“现在还在用的部分 + 它约束了什么”。

## #77 翼身接触迹与 Γ 支撑

它其实是**两个独立缺陷**，修在同一个 PR 序列里：

### 缺陷 1：Γ 支撑与真实后缘不一致

`conv_s075` 的机身接触区间在 y=0 / 0.2，而真实外露主翼后缘从 y=0.249996 起。
旧 loader 把导出的每个站位都建成独立 Γ 基，于是 y=0/±0.2 上出现“壁面影响≈0”的独立
自由度，而 TE/Kutta 约束仍然作用在它们身上。约束最小二乘仍满秩、壁面残差也小，但三个
近零方向被强烈激发：

    cond(B@N) ≈ 4.7e5, ||mu|| ≈ 228, ||gamma|| ≈ 298, CL ≈ 1328

修法：从**声明的上/下 TE 面对**推导真实半展支撑，剔除内侧接触站位；只有当导出的
`gamma_station_dof` 明确把该站位绑到真实 TE 根 DOF 时才剔除，否则直接 `ValueError`。
尾迹几何不变（y=0/0.2 的接触节点仍参与插值），镜像后的两个后缘根站都保留。

### 缺陷 2：接触迹上的势跳变

真正的病因是 `smooth_wall_contact`（尾迹/机身接触曲线）在 P1 适配器里**没有分裂势
DOF**：两片壁面共用一个 μ，于是非零的尾迹势跳变被挂在一条人为连续的壁面迹上，接触处
的近场梯度与压力不物理（旧表 `body max|Cp| = 15.47`、`CDp = 0.0999`）。

同时发现根部 Kutta 采样按“同岸最近质心”选到了机身三角形（tag 20），必须改用声明的
TE 岸面。

修法：对每条声明的 cut 边要求“两个反向绕的壁面 + 一个尾迹边界面”，只沿 cut 边分裂
顶点扇；加一行

    mu_plus - mu_minus = H_contact * Gamma

其中 H 用接触顶点上的**精确**尾迹插值；TE 与接触共享的 junction 点只写一条 jump 行；
**不加**压力/速度 Kutta 行。原来的 46 条 TE Kutta、18 个 Γ 不变，conv 多 67 个接触迹
DOF 与 67 条兼容行。根部不再强制零 Γ，只有外翼尖是零 Γ 端点。双岸约定按边定向，
不按 normal_z 符号（有旋转不变性 oracle）。

### 现在还在用的（代码落点）

- `panel/p1_geometry_contact.py`、`panel/p1_wake_contact.py`：接触/cut 声明与分类；
- `panel/p1_lifting_g1.py` / `panel/p1_lifting_g2.py`：cut 边分裂、接触跳变行、
  声明式 TE 岸面采样（压力 Kutta 的前提）；
- `solution_status` 灾变门：非有限、|CL|>20、|CDp|>1、||γ||>50||μ||、条件数>1e6、
  近零支撑 Γ 列 → `implausible_solution`。它是**灾难门，不是精度验收**。

### 冻结数字（6224 面 conv_s075，α=5°，单线程）

| 变体 | CL_p | CL_γ | CDp 总 | CDp 机身 | 机身 max\|Cp\| | 条件数 |
|---|---:|---:|---:|---:|---:|---:|
| 只分裂 TE（旧） | 0.406014 | 0.411429 | 0.099855 | 0.095938 | 15.475 | 353.07 |
| 接触分裂 + 旧采样 | 0.474670 | 0.479901 | 0.012926 | 0.005777 | 1.05599 | 306.83 |
| 接触分裂 + 声明 TE 采样 | 0.474647 | 0.479859 | 0.012924 | 0.005775 | 1.05598 | 308.95 |
| 同算子 pressure Kutta | 0.465828 | 0.470903 | 0.012167 | 0.005670 | 1.05159 | 不可比 |

第二行到第三行说明采样是二阶修正；第一行到第二行是病根。相关套件 83 passed。

### 遗留

接触跳变兼容不等于物理精度：quadrature/grid 收敛、更多接触 CAD 模型都没做；
`CDp ≈ 0.0129` 是“在这个模型上可解释”，不是“验证过的阻力”。

## #92 Torch 移植

### 做了什么

全部生产求解从 NumPy/SciPy LAPACK 迁到 Torch CPU，分切片 S1–S12 完成：

- 先做 AST 可达清单（不是 grep）：三个种子模块可达 14 个模块，45 处 `np.linalg`/scipy
  调用，真正的决策点只有 2 个 `least_squares`、6 个 `null_space`、5 个 `lstsq`、
  3 个 `svd`、2 个 `qr`、2 个 `solve`、2 个 `solve_triangular`；求值器模块零 LAPACK。
- S9 默认值转向 torch；S10 NumPy 链退役（冻结为 oracle，tag `numpy-path-frozen-s8`）；
  S11 活跃路径不再持有 NumPy LAPACK（2026-11-03）。

### 现在还在用的（代码落点）

- `panel/p1_rank_aware_torch.py`：秩感知最小二乘/零空间（`rcond=1e-11` 的判定是移植中
  唯一的真风险点）；
- `panel/p1_pressure_kutta_torch_driver.py`：结构化 Gauss-Newton + B 预分解 + Woodbury；
- `panel/p1_potential_jump_torch.py`：仿射状态 + Newton；
- `panel/p1_te_recovery.py`：后缘速度恢复（见 [TRAILING_EDGE.md](TRAILING_EDGE.md)）；
- 冻结的 NumPy 结果包在 `verification/issue92/references/`，只作 oracle（误差约 1e-6）。

### 代价与教训

- **同一进程里 torch 之后再调 SciPy LAPACK 会 0xC0000005**。证据脚本必须分开进程跑，
  这条教训写在 `p1_lifting_g1.py` 的 `_torch_null_space` docstring 里。
- 测试期隔离而不是生产 worker：torch 线程数与 BLAS 线程数都固定为 1，否则结果不可比。
- 移植不改变物理模型；它只改“用哪个后端解同一个方程”。

### B3 与 VSPAERO

`docs/archive/issue/ISSUE92_B3_VSPAERO_PROTOCOL.md` 是自由尾迹/B3 与 VSPAERO 的对照协议（600 行），
`vspapi/` 是仓库里唯一的 OpenVSP/VSPAERO 客户端（见 [VSPAPI.md](reference/VSPAPI.md)）。
**VSPAERO 不是真值 oracle**，它只提供同几何的外部读数；`physical_accuracy_accepted`
仍是 false。

## 还开着的口子

- #113：自由尾迹在 swept30/椭圆/BWB 上的“CL 崩 77%”未在现网格复现，需在
  `potential_jump` 默认下按 E1–E4 协议重测（见 [TRAILING_EDGE.md](TRAILING_EDGE.md) §6）。
- #115：标模校准链（C01 圆柱 → C06 NACA0012），`benchmarks/calibration/reference.json`。
- #103：P1 几何可微的边界（顶点级 VJP 未做，见 [ROUTE.md](ROUTE.md) §3）。
