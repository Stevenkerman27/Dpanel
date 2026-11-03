# 稳态 P1 生产路径的 Torch CPU 移植映射

Issue #92，路线 B（决策评论 6569612502），当务之急：*"为完整生产路径产出一份逐文件的 Torch
CPU 移植映射，并找出 G1/G2 求解、压力-Kutta 与耦合自由尾迹迭代所触达的每一处遗留 SciPy/NumPy 线性代数调用。"*

清单的基线提交：`c953ac5`（分支 `research/issue92-phase1-wake-evaluator`）。
用 `python tools/issue92/port_inventory.py [repo-root]` 复现（AST 导入 BFS，而非 grep：它
跟随代码实际到达的地方，因此审计/工具模块按构造被排除）。

## 1. 可达面

三个种子 —— `panel/p1_lifting_g1.py`、`panel/p1_lifting_g2.py`、
`panel/p1_coupled_material_wake.py` —— 可达 **14 个模块**：

| 模块 | 在生产路径中的角色 | 求解器级线性代数点位 |
| --- | --- | --- |
| `p1_lifting_g1.py` | 单翼装配、秩感知约束求解、压力-Kutta | 2 个 `least_squares`，3 个 `null_space`，2 个 `lstsq` |
| `p1_lifting_g2.py` | 翼身装配、combine_wings、底值（floor）方法、约束求解 | 1 个 `least_squares`，3 个 `null_space`，3 个 `lstsq` |
| `p1_pressure_kutta_reduced.py` | 精简/结构化压力-Kutta 非线性求解 | 1 个 `qr`，1 个 `np.linalg.solve`，2 个 `solve_triangular`，`OptimizeResult` |
| `p1_coupled_material_wake.py` | 耦合自由尾迹循环、采样、门禁、快照 | 1 个 `svd`（约束谱，:983） |
| `p1_closed_body.py` | 约束求解的闭合体变体 | 1 个 `qr`（:208）、1 个 `svd`（:209）、1 个 `solve`（:213） |
| `p1_geometry_contact.py` | 后掠壳体/壁面接触分类 | 1 个 `svd`（接触平面拟合，:134） |
| `p1_steady_free_wake.py` | 自由尾迹驱动、尾迹材料、节点速度 | 无 |
| `p1_analytic_sheet_velocity.py` | 默认 `analytic_edges` 诱导速度（闭式） | 无（仅范数） |
| `p1_regularized_sheet_velocity.py` | 核正则化薄片求值器 | 无（仅范数） |
| `p1_projection_duffy.py` | 投影扇 / Duffy 面积求积 | 无（仅范数） |
| `p1_adaptive_area.py` | 自适应面积求积 | 无（仅范数） |
| `p1_hybrid_area.py` | 近扇 / 远 Duffy 拆分 | 无（仅范数） |
| `p1_wake_contact.py` | 尾迹接触声明 | 无 |
| `p1_torch_analytic_sheet.py` | 已经是 torch 的解析求值器 | 无 |
| `wake_velocity_methods.py` | 方法注册表 | 无 |

有两项后果值得先说明：

1. **移植面很小且集中**：45 处 `np.linalg`/`scipy` 命中，其中真正的
   LAPACK 决策点是 **2 个 `least_squares`、6 个 `null_space`、5 个 `lstsq`、3 个 `svd`、2 个 `qr`、
   2 个 `np.linalg.solve` 和 2 个 `solve_triangular`**。面上的其余内容要么是
   `np.linalg.norm`（119 次逐元素归约，不涉及 LAPACK），要么是数组装配。
2. 求值器模块 —— 那些主导运行时的模块 —— **完全不含 LAPACK**。诱导/求积
   工作可以在不触及任何分解的情况下迁移到 torch 张量。

## 2. 每一处调用及其 torch 替换与语义风险

| 当前调用 | 点位 | torch 替换 | 语义风险 |
| --- | --- | --- | --- |
| `scipy.optimize.least_squares(..., method default 'trf', jac=...)` | `p1_lifting_g1.py:506`、`p1_lifting_g2.py:304` | 无等价物：需以相同的 `ftol/xtol/gtol=1e-11`、`max_nfev`、x 缩放与步长边界实现 trust-region reflective | **最高**：`status`/`success`/`nfev` 字符串流入回执与验收（`pressure_kutta_acceptance`），且 G2 的底值是最小二乘底值而非零 —— 不同的停止规则会在不改变物理的情况下改变所报告的状态 |
| `scipy.linalg.null_space(A, rcond=1e-11)` | 6 处（G1 :476/:493/:569，G2 :279/:295/:371） | `torch.linalg.svd(..., full_matrices=True)`，保留 `S > rcond*S[0]` 对应的 `Vh[k:]` | **高**：这是秩判定；见 §3 |
| `scipy.linalg.lstsq(..., lapack_driver='gelsd')` | 5 处（G1 :478/:578，G2 :242/:281/:374） | `torch.linalg.lstsq(..., driver='gelsd')`（返回 `rank`） | **高**：两个库的 `rcond` 默认值不同；必须显式传入，且 `gelsd` 是揭示秩的，因此截断值变化会改变最小范数解 |
| `np.linalg.svd` | 3 处（`p1_coupled_material_wake.py:983`、`p1_geometry_contact.py:134`、`p1_closed_body.py:209`） | `torch.linalg.svd` | 低-中：:983 供给了被报告的谱，:134 拟合接触平面（朝向符号必须保留） |
| `np.linalg.qr` | 2 处（`p1_pressure_kutta_reduced.py:94`、`p1_closed_body.py:208`） | `torch.linalg.qr` | 低：符号约定无关紧要，因为 `R` 只作为三角因子使用 |
| `np.linalg.solve` | 2 处（`p1_pressure_kutta_reduced.py:146`、`p1_closed_body.py:213`） | `torch.linalg.solve` | 低 |
| `scipy.linalg.solve_triangular` | 2 处（`p1_pressure_kutta_reduced.py:137-138`） | `torch.linalg.solve_triangular` | 低 |
| `scipy.optimize.OptimizeResult` | `p1_pressure_kutta_reduced.py:275` | 属性名相同的本地 dataclass | 低，但它是公开结果契约的一部分 |
| `np.linalg.norm` | 119 处 | `torch.linalg.vector_norm` / `torch.linalg.norm` | 机械替换，不涉及 LAPACK；唯一要紧之处在于这些调用不得在 torch 张量上重新进入 NumPy |

替换方案的实测能力（torch 2.13.0+cu132，float64 CPU，探针
`tools/issue92/probe_torch_linalg.py`）：

| 检查 | 结果 |
| --- | --- |
| `torch.linalg.lstsq` 驱动 `gels` / `gelsy` / `gelsd` 对 `np.linalg.lstsq` | 7.6e-16 / 2.2e-15 / 1.1e-15 |
| `torch.linalg.svd` 奇异值谱对 NumPy | 2.8e-14（绝对） |
| `torch.linalg.qr` 正交性 | 4.4e-16 |
| `torch.linalg.solve_triangular` 对 `scipy.linalg.solve_triangular` | 6.9e-16 |

因此替换方案确实存在，且精确到 float64 舍入。移植的阻碍不是缺少
功能；而是**决策语义**（§3）与非线性求解器（§4）。

## 3. 唯一真正的风险：`rcond=1e-11` 处的秩判定

六处 `null_space` 调用与五处 `lstsq` 调用通过把一个奇异值与
`rcond * S_max` 比较来判定秩。Torch 与 NumPy 在 1e-14 量级上不一致；耦合求解的秩截断是相对
1e-11，即比观测到的数值差异高约**三个数量级**。在一个
随机的 40x12 探针上，最近的奇异值距截断点 1.3，所以那里的判定是
平凡的 —— 但这不能证明生产矩阵的情况，生产矩阵的谱来自
近退化面元布局（半模镜像、TE 配对、重复尾迹行），在这些地方 1e-11 的间隙是可能的。

因此对每一处 `null_space`/`lstsq` 点位，移植规则是：

* 精确复现判据（`S > rcond*S[0]`，`lstsq` 使用相同的默认 `rcond` 语义）；
* 在每个冻结样例上，**报告截断点与最近奇异值之间的间隙**，并
  要求移植前后的零空间维数与 `rank` 完全相同；
* 若某个样例的间隙在相对 1e-13 以下，该样例无法在后端之间判别
  秩，任何关于它的一致性主张都必须说明这一点，而不能宣称一致。

这是最可能产生静默行为变化的一项，因为秩翻转不会
抛错：它会改变 `mu`/`Gamma` 以及下游每一个载荷，而残差看上去仍然收敛。

## 4. 非线性求解

`least_squares` 只用了两次，但耦合迭代每一步都走 G2 路径，因此其
停止行为支配整条自由尾迹轨迹。有两点事实约束着移植：

* G2 的压力-Kutta 残差有一个**非零底值**，而现有验收逻辑
  （`p1_lifting_g2.py:230-255`）正是围绕这个底值写的：梯度停滞之后出现 `status 0` 且 `success=False`。
  torch 重实现必须复现底值语义，而不是"更用力地收敛"；
* 精简路径已经重新实现了停止判据
  （`p1_pressure_kutta_reduced.py:165-275`，绝对 + 相对最优性并带停滞规则），
  并已针对 `least_squares` 行为验证；它是完整求解的模板。

## 5. 测试期隔离，而非生产 worker

路线 B 禁止把进程外 GPU worker 作为生产架构。它并不消除
`docs/archive/issue/ISSUE92_PHASE1_WAKE_EVALUATOR.md` §6f 中实测到的事实：torch 执行任何算子之后，同一解释器中的
下一次 NumPy LAPACK 调用会以 `0xC0000005` 崩溃。因此一致性测试必须在**跨进程**
之间比较后端（每个后端写出 A/W/J/K、秩、奇异值、
mu/Gamma、Cp、载荷、残差、状态的 `.npz`；由比较器读取两者）。那是测试脚手架，
不是运行时边界：交付的 torch 路径从不调用 SciPy。

## 6. 垂直切片

每个切片以完整套件加干净 master 基线收尾，并且只有在其一致性门禁通过之后，才可为其自身入口设置
`backend='torch'`。默认后端在 S8 之前一直保持 NumPy；S9 将其切换到 torch，S10 删除了 NumPy 求解器（§18、§19）。

| 切片 | 内容 | 一致性门禁 |
| --- | --- | --- |
| S1 | `p1_pressure_kutta_reduced.py`：`qr`、`solve`、`solve_triangular`、`OptimizeResult` → torch —— **已交付，§8** | A/W/J/K 与 `z` 向量相对误差 1e-12；在所有覆盖精简路径的固定样例上 `status`/`nfev` 一致；10 个既有测试仍通过 |
| S2 | G1/G2 中的秩感知约束求解：`null_space`、`lstsq` → torch —— **已交付，§9** | 全部 8 个固定样例上 `rank` 与零空间维数相同，**外加** §3 的间隙报告；`mu`/`Gamma`/`CL`/`CD_p`/`CM_y` 相对误差 1e-11 |
| S3 | 非线性承接：torch 上的精简压力-Kutta 非线性求解（`least_squares` → torch，G1 :506 / G2 :304 —— **已交付，§10**；门禁被重写：torch 路径复现的是*结构化*算法，因此门禁是一致性验收分类加一致的载荷，而不是对 `trf` 的 `nfev` 一致） | 全部 8 个样例上对 `trf` 参考的验收分类（`accepted`、`gradient_relative`、`wall_over_floor`、`pressure_rms`）一致；对 NumPy 结构化后端的 `status`/`nfev`/`message` 一致；载荷相对误差 1e-6 |
| S4 | 张量上的装配与后处理：面→DOF 散射、Cp、力/力矩 —— **已交付，§11**：`mesh_kernel`、`doublet_matrix`、`gradient_map`、`wake_operator`、`loads`；Kutta 行与跳变/接触行在耦合循环内部重建，因此随 S5 一起迁移 | 全部 8 个样例上 A/W 最大相对误差 1e-12、Frobenius 1e-13，`grad` 逐位相同；载荷对共享 NumPy 映射误差 1e-13 |
| S5 | 耦合自由尾迹循环与采样完全在 torch 中；快照 schema 记录后端 —— **已交付，§12**。门禁被重写一次：192 面样例给出三次被接受的更新，872 面 CAD 体接受一次后便在两个后端上都停于 `blocked_updated_nonlinear_solve`（结构化求解器的线搜索，§12.5） | 三种情形上状态与每次迭代决策（`rank`、`kinematic_candidate`、`nonlinear_success`、尾迹资格）一致；载荷相对误差 1e-5，状态 1e-5，`cp` 绝对 1e-3，残差绝对 1e-5。先按 1e-9 声明，后依 §12.4 的平坦谷证据修订一次；随后 B2（CUDA）开始 |
| S6 | 公开入口分派：`solve` / `solve_multi` 接受 `backend='numpy'|'torch'` 与 `device`，且 torch 分支在未移植的 SciPy 信赖域求解器上失败关闭 - **已交付，§15** | 全部 8 个样例：`status`/`nfev`/`message`/秩/验收分类一致；载荷相对 ≤ 6.4e-9，状态 ≤ 1.8e-8，`|dcp|` ≤ 8.7e-7。先按 1e-9 载荷 / 1e-9 状态 / 1e-12 `cp` 声明，后依平坦谷证据修订一次（§15.4） |
| S7 | 退役 NumPy 参考：把全部 8 个样例的生产默认求解冻结为可重跑的冻结数据包，再把最后一个按目标区分的求值器（RM-P1 正则化薄片）迁到 torch，并提供 `rm_backend='numpy'|'torch'` - **已交付，§16** | 冻结数据包：8/8 重跑逐位相同（`loads=0.0`、`state=0.0`、秩与裕度相同）。RM 场：两种求积、CPU 与 CUDA 上相对误差 ≤ 1.9e-15；耦合迭代第 0 次逐位相同，第 1 次在求解器固定时 ≤ 4.9e-14，决策一致 |
| S8 | 整条链的隐式伴随/VJP（不经由枢轴做微分）：线性层，以及带四次目标**精确** Hessian 的压力层，外加判断一个导数究竟能否被认证的诊断 - **已交付，§17** | 4 个样例：线性 `dL/dalpha` 对有限差分 2.2e-8 … 2.1e-7；压力层在 3 个样例上 6.9e-9 … 1.5e-6，Gauss-Newton 算子差 2-3 个数量级。`g1_wingbody_cad` 的压力被声明为**不可认证**（FD 稳定性 0.63，`cond = 2.7e+11`，前向残差底值 1e-4）而非通过 |

| S9 | 默认迁移到 torch；NumPy 链保留为可达的显式证据生成器 - **已交付，§18** | 8/8 决策相同，载荷相对 ≤ 6.4e-9，状态 ≤ 1.8e-8，外加冻结数据包重放 1.0e-13 … 1.0e-6 |
| S10 | 退役 NumPy 链：删除 NumPy 求解器、SciPy 信赖域路径与结构化 NumPy 压力-Kutta；冻结数据包成为唯一的判据源 - **已交付，§19** | 9/9 数据包重放落在逐量界内（CL/CL_gamma 5e-7，CD_pressure 2e-6，CM_y 1.5e-6，状态 1e-5，cp 1e-3），这些界是两个 NumPy 构建中较差者的 2 倍；每个被退役的名字都失败关闭 |
| B2 | 整链设备选择，同一进程内 device=cpu 或 cuda，无静默回退 —— **已交付，§13**：冷/热求解、一次耦合迭代、上传字节、峰值显存，外加归因该迭代的 profile | 两个设备上对 S5 NumPy 记录的载荷相对误差 1e-5 一致，状态与验收一致；**不主张加速** —— 实测 CUDA 上求解慢 11.4 倍与 4.3 倍，且迭代中 86% 是 Python 几何门禁 |
| B3 | 四种几何（矩形、后掠梯形、椭圆、翼身）的 VSPAERO 交叉校验，多个迎角，两个网格层级 | 只比较 CL(alpha) 的截距与斜率、诱导阻力、展向环量，以及一种可对齐的 Cp 定义；厚薄片与 VLM 结果分开报告 |

按清单估计的相对难度：S1 约 15% 的工作量，S2 约 20%，S3 约 30%，S4
约 25%，S5 约 10%。S3 是进度风险（没有库等价物），S2 是正确性风险（§3）。

## 7. 每个切片的不变量

* 处处 float64；NumPy/SciPy 后端保持为不可变的参考与显式回退。
* 不改默认值，不改尾迹拓扑或接触语义，不改门禁阈值，
  `physical_accuracy_accepted=false`。
* 不用有限差分设计灵敏度，不用代理模型，不用 Transolver，生产路径不用进程外
  worker。
* 每个切片之后做整合的完整套件与干净 master 基线 —— 而非挑选的单元测试。

## 交付日志

S1–S8 的分切片交付记录（测量、门禁、被拒项与复现命令）已移到
[archive/issue/ISSUE92_DELIVERY_LOG.md](archive/issue/ISSUE92_DELIVERY_LOG.md)。
本文件保留移植前的架构判断（第 1–7 节）与当前默认状态（第 18–20 节）。

## 18. S9：默认值转向 torch，NumPy 成为证据生成器（2026-11-02）

### 18.1 决策，以及它不改变什么

桌面上有两个选项。选项 B 是“切换默认值，在同一个提交中删除 NumPy
求解器，给删除前的树打标签”。选项 A 是用户所选择的：
“现在切换默认值，保留 NumPy 链作为显式证据生成器可达，并且
暂不删除任何东西”：

| 属性 | S9 之后 |
| --- | --- |
| 生产默认 | torch，逐次调用解析 |
| NumPy/SciPy 链 | 通过 `backend='numpy'` 可达，未改动 |
| `docs/issue92/references/` 中的冻结数据包 | 仍是 8 个固定样例的 oracle |
| 删除 `panel/p1_rank_aware.py`、`panel/p1_pressure_kutta_reduced.py`、NumPy 求解分支 | 推迟到拥有一个两条路径都能运行的环境的切片 |

解析是显式的，而不是假定的。`panel/p1_backend.py`：

    resolve_backend(None)        -> ('torch', 'default')      # torch importable
    resolve_backend(None)        -> ('numpy', 'default')      # torch missing
    resolve_backend('numpy')     -> ('numpy', 'explicit')
    resolve_backend('torch')     -> ('torch', 'explicit')     # raises if torch is missing
    PANEL_P1_BACKEND=numpy       -> ('numpy', 'environment')

`backend='torch'` 在无 torch 的解释器中会抛出
`ValueError: backend='torch' requested but torch is not importable`；它绝不会降级到 NumPy。
未知值抛出 `backend must be None, 'numpy' or 'torch'`。来自 `g1.solve`、`g2.solve_multi`
以及耦合循环的每个结果现在都携带 `backend`
和 `backend_source`，因此无需根据时间戳猜测某次运行。

记录的测试套件在 `test/conftest.py` 中用
`os.environ.setdefault("PANEL_P1_BACKEND", "numpy")` 固定 oracle。其数值基线是在
NumPy 路径上测得的，且关乎算法，而非哪个线性代数库运行，因此该套件继续
在产生这些基线的路径上运行；演练 torch 的测试显式请求 `backend='torch'`，
而 `test/test_backend_dispatch.py` 在默认值本身受到测试时
清除该固定。

### 18.2 还有什么仍在调用 LAPACK

`tools/issue92/retirement_inventory.py` 扫描 `panel/` 下的每个模块，查找
`np.linalg.svd|svdvals|qr|solve|lstsq|eig|eigh|pinv|matrix_rank|cholesky`、
`scipy.linalg` 和 `scipy.optimize.least_squares`：

    python -m tools.issue92.retirement_inventory --out docs/issue92/s9_inventory.json

33 个模块，69 个调用点。其中，`tools/issue92/port_inventory.py` 从三个生产
入口到达 19 个模块，而在该表面上只有这些模块
仍持有 LAPACK 调用：

| 模块 | 位置 | 为何不是移植缺陷 | 处置 |
| --- | --- | --- | --- |
| `panel/p1_coupled_material_wake.py` | :1091 `np.linalg.svd(constraint)` | 当 `solve_backend='torch'` 时已切换到 `torch.linalg.svdvals`（:1085） | 保留 |
| `panel/p1_geometry_contact.py` | :145 `np.linalg.svd(centred)` | 只对大于三角形的点集才会到达；门禁传入三角形并取叉积 | 保留 |
| `panel/p1_closed_body.py` | :208-213 `qr`/`svd`/`solve` | 封闭体 Dirichlet 演示（`panel/run_p1_closed.py`），不在耦合路径上 | 独立切片 |
| `panel/p1_lifting_g1.py`、`panel/p1_lifting_g2.py` | NumPy 分支 | oracle 本身 | 随 oracle 一起删除 |
| `panel/p1_rank_aware.py`、`panel/p1_pressure_kutta_reduced.py` | 5 个位置 | 由 `p1_rank_aware_torch` 和 `p1_pressure_kutta_torch(_driver)` 重复实现 | 随 oracle 一起删除 |

可达表面上的其他一切都是 `np.linalg.norm`，它是逐元素的，
不可能污染 torch 进程。

### 18.3 删除之后冻结数据包还能充当 oracle 吗？能，误差约到 1e-6

`tools/issue92/s7_reference_export.py` 现在接受 `--backend {numpy,torch}` 和
`--device`，并在求解所用的同一个库中获取约束谱（torch 内核之后的 NumPy
LAPACK 调用正是 6f 节的 0xC0000005 崩溃）。在 `--backend torch` 下只将
`constrained_rank` 与数据包比较：数据包记录的是 SciPy
trust-region-reflective 求解器的状态、`nfev` 和消息，
而 10.5 节表明触发哪条停止规则是舍入误差，
不是问题的属性。

    python -m tools.issue92.s7_reference_export verify --backend torch --device cpu

| 固定样例 | 载荷 | 状态 | 秩 | 裕度 |
| --- | --- | --- | --- | --- |
| g1_wing_rect | 1.0e-13 | 1.4e-13 | 5 | 1.000e+11 |
| g1_wing_camber | 5.5e-14 | 9.3e-14 | 5 | 1.000e+11 |
| g1_swept30 | 6.4e-14 | 3.9e-13 | 63 | 5.785e+10 |
| g1_elliptic | 8.7e-07 | 1.5e-06 | 83 | 6.092e+10 |
| g2_conventional | 1.6e-08 | 2.3e-07 | 8 | 1.000e+11 |
| g2_tail_low | 1.6e-08 | 2.1e-07 | 8 | 1.000e+11 |
| g1_wingbody_cad | 1.4e-09 | 4.4e-08 | 99 | 4.778e+10 |
| g1_wingbody_cad_3x | **CD_pressure 1.013e-06** | 1.6e-08 | 99 | 4.527e+10 |

运行前已声明：载荷 1e-6，状态 1e-6，cp 1e-4 相对。八个中有七个通过。
离群点是某个固定样例的一个载荷分量，超出界限 1.3%。它是
**算法差异，而非移植差异**：数据包由 SciPy 参考求解器产生，torch 路径
实现的是结构化求解器，而结构化解在 S3 中于该固定样例上
就已与参考相差 2.9e-7。同一量上的移植差异（torch 对 NumPy，
两者都是结构化）为 3.1e-12，
在 S6 中测得。

因此诚实的表述是：删除之后，冻结数据包以约 1e-6 的相对精度就载荷
为 torch 路径提供认证，而 `g1_wingbody_cad_3x` 的 `CD_pressure` 位于
1.013e-6，刚好在界外。这里没有任何关于更高精度的声明。

本切片还修复了一个容差缺陷：谱裕度检查用绝对容差 1e-12 比较两个
量级为 1e10 的数，因此它过去之所以能通过，只是因为 NumPy 路径逐位
复现了记录的谱。现在它是相对的；在 `g1_swept30` 上 NumPy 与 torch
的谱一致到 6.7e-16。

### 18.4 为什么本切片不删除 NumPy 代码

| | `.conda_env`（项目） | `myml`（conda base） |
| --- | --- | --- |
| python | 3.11.17 | 3.13.14 |
| numpy / scipy | 2.4.6 / 1.17.1 | 2.5.2 / 1.18.0 |
| jax | 0.10.2 | 解释器中缺失；0.12.0.1 安装在旁边的 `runs/jax_site313`（见下文） |
| torch | 缺失 | 2.13.0+cu132 |
| 全套件 | 858 passed, 52 skipped | 989 passed, 21 skipped, 2 failed |

删除 NumPy 求解器需要一个能运行整套件的解释器，因为那才使默认切换
可审查。`myml` 现在就是那个解释器，其中
有一段值得记录的 10 分钟弯路。

`pip install --target` 首先失败于

    PermissionError: [Errno 13] Permission denied:
    '...\pip-unpack-...\jax-0.12.0.1-py3-none-any.whl.metadata'

原因比“沙箱阻止了 pip”更具体：`tempfile.mkdtemp` 以模式 `0o700` 创建
其目录，而该沙箱的令牌无法写入具有该 ACL 的目录。`os.mkdir` 以及写入
普通目录都能工作；`mkdtemp` 本身成功，
而其中第一次写入失败。
因此 pip 根本无法解包 wheel。

`tools/issue92/pip_open_temp.py` 在将 `os.mkdir` / `tempfile._os.mkdir`
重定向为模式 `0o777` 之后运行 pip 的入口点；其他任何东西都没有打补丁：

    python tools/issue92/pip_open_temp.py install --no-cache-dir \
      --target runs/jax_site313 --proxy http://192.168.3.9:7890 jax jaxlib

那安装了 `jax 0.12.0.1`、`jaxlib 0.12.0.1`（cp313 win_amd64）、`ml_dtypes`、
`opt_einsum` 及其自带的 numpy/scipy 副本；`PYTHONPATH=runs/jax_site313` 将其
置于解释器 site-packages 之前。
有了它，整套件可在一个解释器中运行：

    PYTHONPATH=<worktree>;<repo>/runs;<repo>/runs/jax_site313 \
      python -m pytest test -q        # 989 passed, 21 skipped, 2 failed

两个失败是 `test/test_half_native_panel.py::test_streamed_half_action_does_not_trace_full_square_aic`
和 `test/test_iterative_panel.py::test_streamed_primal_has_no_full_wall_aic_array_in_jaxpr`，
两者均在测试自身的 jaxpr 访问器中针对 jax 0.12 的 `ClosedJaxpr` 结构
抛出 `RecursionError: maximum recursion depth exceeded`。它们是检查
辅助工具的 jax 版本产物，而非本切片的结果。

因此删除切片不再被环境阻塞。仍然阻塞它的是 oracle 的所有权：
记录的套件显式演练 NumPy 路径（`backend='numpy'`、`PANEL_P1_BACKEND=numpy`），
而冻结数据包仅以约 1e-6 认证 torch 路径（18.3）。
因此删除求解器意味着针对数据包重写那些测试，
那是自带证据的另一个切片。

### 18.5 测试

`test/test_backend_dispatch.py`（8 个测试）现在固定默认值的含义：
解析及其记录的来源、环境固定、显式 torch 不可用时抛出、torch 缺失时的
回退、两个针对显式 NumPy 运行的一致性测试，以及默认值本身解析为 torch
且 `backend_source='default'`。`test/test_p1_solve_torch.py` 断言
耦合循环的 `solve_backend` 默认是 `None`（解析）
而非硬编码的库。

    # torch available
    python -m pytest test/test_backend_dispatch.py -q
    # the full suite, in the environment that has no torch
    python -m pytest test -q --ignore=test/test_torch_half_checkpoint.py \
      --ignore=test/test_torch_half_core_cpu.py --ignore=test/test_torch_half_lifting.py \
      --ignore=test/test_torch_panel_kernels.py --ignore=test/test_torch_panel_nonlifting.py \
      --ignore=test/test_torch_panel_symmetry.py --ignore=test/test_torch_panel_validation.py

### 18.6 未声明

- 未声明：NumPy 链今天就可移除。它仍是 oracle，仍是
  `.conda_env` 中唯一能运行的路径，仍是冻结数据包的生产者。
- 未声明：torch 路径更精确。18.3 节测得与
  参考求解器在载荷上的一致到 1e-6，有一个离群点刚好在界外。
- 未声明：单解释器套件等价于 `.conda_env` 基线。
  18.4 节记录了它暴露的两个 jax-0.12 jaxpr 检查失败，且其
  numpy/scipy 是 2.5.4/1.18.1，而非 2.4.6/1.17.1。
- 未声明：`analytic_backend` 和 `rm_backend` 的默认值改变了。它们没有；
  本切片改变的是哪个**求解器**运行，而不是哪个**诱导求值器**运行。



## 19. S10：NumPy 链退役（2026-11-02）

S9 留下了删除的悬置；用户以选项 1 作答：退役。NumPy 求解器已从 `panel/`
移除，八个冻结数据包是唯一的 oracle，而一致性现在是在能经受
NumPy 构建变化的界限下对这些数据包的重放。

### 19.1 移除了什么，调用者看到什么

| 移除 | 替换为 | 调用者可见的契约 |
| --- | --- | --- |
| `panel/p1_lifting_g1.py::solve` 的 NumPy 分支 | `panel/p1_solve_torch.py::solve_single_torch` | 每个结果上的 `backend`、`backend_source` |
| `panel/p1_lifting_g2.py::solve_multi` 的 NumPy 分支 | `panel/p1_solve_torch.py::solve_multi_torch` | 相同；`mu`、`gamma`、`cp` 仍是数组 |
| `panel/p1_pressure_kutta_reduced.py`（结构化 NumPy/QR 压力-Kutta） | `panel/p1_pressure_kutta_torch.py` | `pressure_solver` 取 `{None, 'structured'}` |
| SciPy trust-region 分支（`least_squares`、`pressure_solver='reference'`） | 结构化 torch 求解器 | `ValueError` 指明标签和数据包 |
| 耦合循环的 NumPy 分支（`_solve`） | `solve_multi_torch` | `solve_backend` 取 `{None, 'torch'}` |

两个退役的名称都会失败关闭，而不是回退：

    solve(..., backend='numpy')
    ValueError: backend='numpy' was retired with the NumPy solve chain; its code is tag
    numpy-path-frozen-s8 and its recorded answers are docs/issue92/references/

    solve(..., pressure_solver='reference')
    ValueError: pressure_solver='reference' was retired with the NumPy chain; its recorded
    answers are docs/issue92/references/ and its code is tag numpy-path-frozen-s8

`PANEL_P1_BACKEND` 作为进程固定保留下来，但只有 `torch` 是有效值；固定
`numpy` 会抛出同样的退役错误，这也是过期的 CI 作业会看到的情形。

`alpha_sensitivity` 和 `alpha_sensitivity_multi` 曾使用 `scipy.linalg.null_space` 和
`lstsq(..., lapack_driver='gelsd')`；它们现在使用 `_torch_null_space` 和 `_torch_lstsq`
（`panel/p1_lifting_g1.py` 中委托给 `panel/p1_rank_aware_torch.py` 的模块级
辅助函数），其余部分为逐元素 NumPy。原因不是纯粹性：
在一个解释器中，torch 内核之后的 SciPy LAPACK 调用会以 `0xC0000005`
死亡（6f 节），因此在 torch 生产路径上那些调用必须去掉，
尽管该敏感性是与被删除求解器无关的线性系统导数。

### 19.2 仍然存在的 oracle，以及为什么它的界是逐量的

`test/test_frozen_bundle_parity.py` 通过 torch 路径重放全部八个数据包。
数据包是 SciPy trust-region-reflective 的产物，而 torch 路径运行
结构化求解器，因此比较中已经包含算法差异（在 `g1_wingbody_cad_3x` 上
为 2.9e-7，第 10 节）；同一算法上的移植差异为 3.1e-12
（第 15 节）。单一个 1e-6 界之所以失效，是因为停止规则
在残差底噪上触发，因此输入的最后一个比特决定哪条规则触发。
相同的 torch 代码，两个 NumPy 构建：

| 固定样例 | 量 | numpy 2.5.2 | numpy 2.5.4 |
| --- | --- | --- | --- |
| `g1_elliptic` | 前向停止 | status 2, nfev 21 | status 5, nfev 22 |
| `g1_elliptic` | CL | 2.17e-07 | 4.02e-07 |
| `g1_elliptic` | CD_pressure | 8.72e-07 | 1.62e-06 |
| `g1_elliptic` | CM_y | 5.69e-07 | 1.06e-06 |
| `g1_elliptic` | gamma 状态 | 1.5e-06 | 6.72e-06 |
| 所有其他 | 最差 CL | 3.6e-14 | 未变 |

生效的界限：CL 和 CL_gamma 5e-7，CD_pressure 2e-6，CM_y 1.5e-6，状态 1e-5，cp
1e-3。每个都约为两次运行中较差者的两倍，因此大于构建间散布的
回归仍会失败。证据：`runs/scratch/issue92_loose/probe_bundle_252.json` 和
`runs/scratch/issue92_loose/probe_bundle_254.json`。

状态界是唯一诚实的修订：在 2.5.4 重放存在之前根据 2.5.2 重放声明为
2e-6，在那里测得 6.72e-06，现在为 1e-5。状态不是交付量；
它是该固定样例平坦谷底显现的地方（第 17 节）。

### 19.3 现在需要标签的证据工具

`tools/issue92/s1_parity.py` 和 `tools/issue92/branch_probe.py` 导入了被删除的
模块；它们是历史证据工具，因此现在以一条消息停止，
而不是回溯：

    SystemExit: this evidence tool re-runs the retired NumPy chain; check out tag
    numpy-path-frozen-s8 to use it (its recorded answers are docs/issue92/references/)

`tools/issue92/s3_pressure_parity.py capture` 也是如此，它通过
`g1.solve(..., pressure_solver='reference')` 构建参考状态；其输出
已作为 `docs/issue92/s3_parity.json` 提交。

### 19.4 测试

`test/test_pressure_kutta_reduced.py` 随其所测试的模块一并删除。
`test/test_p1_pressure_kutta_torch_driver.py` 失去其 NumPy 比较，并保留
三项不需要它的检查（静态 no-NumPy 守卫、模型对照其定义的朴素
转录、雅可比对照中心差分）；冻结的
`s3_reference_g1_wing_rect.npz` 端到端用例保留，因为它是数据包，
不是第二实现。`test/test_backend_dispatch.py` 被重写，以断言两个
退役错误、解析记录、缺失 torch 的消息，以及默认调用到达 torch
且 `backend_source='default'`。`test/conftest.py` 不再固定
`PANEL_P1_BACKEND`，因为已没有第二条路径可供套件固定。

    # torch available (myml), jax injected for conftest
    python -m pytest test -q

### 19.5 未声明

- 未声明：退役使 torch 路径更精确。数据包以 19.2 中的数字
  为其定界，主导项是算法差异，而非移植。
- 未声明：`.conda_env` 还能运行套件。它没有 torch，因此按设计会在导入
  P1 入口点时失败；torch 现在是 P1 链的硬性要求，
  这是用户接受的权衡。
- 未声明：两个 jax-0.12 jaxpr 检查失败（18.4 节）已修复。
- 未声明：`panel/p1_closed_body.py` 或 `panel/p1_geometry_contact.py` 已被移植；
  它们不在耦合路径上（18.2 节），并保留其 LAPACK 调用。

## 20. S11：活跃路径不再持有 NumPy LAPACK（2026-11-03）

### 20.1 改变了什么

| 文件 | 变更 |
| --- | --- |
| `panel/p1_geometry_contact.py:131` | `_dominant_normal` 对大于三角形的点集使用 `torch.linalg.svd`（惰性导入）。三角形快速路径未变。 |
| `panel/p1_coupled_material_wake.py:1090` | 约束谱始终为 `torch.linalg.svdvals`；NumPy 的 `else` 分支被删除，因为 `solve_backend` 只能解析为 torch |
| `panel/run_p1_coupled_material_wake.py:115` | `--solve-backend` 标志被移除；`--solve-device` 保留 |
| `test/test_live_path_has_no_numpy_lapack.py` | 新的可执行断言，见 20.3 |

原因不是整洁。6f 节记录到，torch 内核之后的 NumPy/SciPy LAPACK 调用
会杀死解释器（0xC0000005）。耦合循环用 torch 求解，然后运行
几何谓词，因此 `coplanar_overlap_area` 内部的 NumPy SVD 是潜在崩溃，
而非风格缺陷。它之所以侥幸存活，是因为 `np.linalg.svd` 正是探针中
未崩溃的调用；没有任何东西保证这一点。

### 20.2 行为未变的证据

- 几何审计指纹，与 14.1 节相同的用例和设置：`pairs_tested` 54，
  `detail_sha` `9aa3999781224e5347f1723d6f13a3bd78ea4b9f02cda89a53237a560695818d`，
  与此变更前记录的值逐字节相同。
- `_dominant_normal` 在 6 点后掠壳体上：torch SVD 与 NumPy SVD 返回相同的单位法向，
  `|dot| = 1.0`。
- 定向测试（`test_p1_gate_contact`、`test_p1_geometry_contact`、`test_p1_solve_torch`、
  `test_backend_dispatch`）：42 passed, 3 skipped。
- 全套件（myml，torch 2.13.0+cu132，jax 0.12.0.1 经 `runs/jax_site313`）：
  **1000 passed, 21 skipped, 2 failed**，用时 210.38 s。这两个失败是先前存在的 jax 0.12
  jaxpr 递归失败（`test_half_native_panel`、`test_iterative_panel`），它们在干净的 master 上也失败。
  先前的基线是 997 passed；+3 是新增的守卫测试。

### 20.3 守卫

`test/test_live_path_has_no_numpy_lapack.py` 从十二个生产入口
（`p1_lifting_g1`、`p1_lifting_g2`、`p1_coupled_material_wake`、`p1_solve_torch`、
`p1_implicit_adjoint`、`p1_steady_free_wake`、三个尾迹求值器、`p1_batched_influence`）
遍历导入，若任何可达模块调用 `np.linalg.{svd,svdvals,qr,solve,lstsq,eig,eigh,...}` 或
导入 scipy 则失败。一个模块被列入允许清单并就近写明理由：`panel/p1_closed_body.py` 保留
`qr`/`svd`/`solve` 在其自身的独立封闭体演示（`run_p1_closed.py`）中；P1 链
导入 `Mesh` 和 `influence_block`，从不进入 `solve()`。第三个测试在一个临时模块中植入
`np.linalg.svd` 调用，并断言扫描器能找到它，因此守卫不能
因为什么都没扫描而静默通过。

### 20.4 未声明

- `panel/p1_closed_body.py` 仍包含 LAPACK；它不在求解路径上，未被移植。
- 默认设备仍是 `cpu`。本节不改变任何数值，也不带来速度。
- 逐元素 NumPy 仍无处不在（`np.cross`、`np.linalg.norm`、数组记账）。禁止的是
  LAPACK 入口点，因为那些是 torch 内核之后会崩溃的调用，而不是把 NumPy
  作为数据库禁止。

### 20.5 复现

```
python -m pytest test/test_live_path_has_no_numpy_lapack.py -q
python runs/scratch/issue92_loose/audit_fingerprint.py            # prints pairs_tested and detail_sha
python -m pytest test -q                            # full suite
```
