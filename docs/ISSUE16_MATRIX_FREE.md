# issue #16 本地实测：矩阵无关 GMRES、近场分块预条件、严格镜像半模

日期 2026-10-25。实现位于 `runs/pr13_work/panel/matrix_free.py`（新增，约 380 行），
驱动脚本 `runs/mf_test.py`（一致性审计）、`runs/mf_bench.py`（独立进程基准）、
`runs/mf_adjoint.py`（隐式伴随）、`runs/mf_loads.py` + `runs/mf_load_check.py`（原始载荷对拍）、
`runs/mf_half_vortexad.py`（严格镜像系统上的半模降阶验证）。

所有数字都是本机 CPU 实测：Python 3.11.17、JAX 0.10.2（CPU，x64）、NumPy 2.4.6、
SciPy 1.17.1、32 逻辑核、`PANEL_KERNEL_BLOCK=64`、`TMP/TEMP/TMPDIR=runs/pipwork`。
`physical_accuracy_accepted` 仍为 false，本任务只动求解器与内存，不改物理验收口径。

## 1. 实现

### 1.1 显式 Av / Aᵀv，不形成矩阵

`PanelOperator(inputs, block=64)` 预计算几何与核参数（tri、center、normal、area、sigma、
wake 三角形、hat basis/slope、Kutta 质量矩阵 integrals/measure、upper/lower TE 面号），
三个 `jax.jit` 闭包 `_doublet(targets, rows)`、`_wake(targets)`、`_source(targets)` 负责单块作用。

- `matvec(v)`：按块 `i0:i1` 计算 `doublet_rows`（行切片经 `row_index` 传入，只在本块上出现对角 −1/2 迹），
  立即与 `v[:nf]` 收缩累加到 `out[i0:i1]`，再叠加 `wake_rows @ v[:nf]`，最后写 Kutta 行
  `out[nf:] = -kutta(mu) + mass @ gamma`。整条路径不分配 (nf, nf) 数组。
- `rmatvec(w)`：`out[j] += Σ_i A[i,j] w[i]`，同样逐块累加；Kutta 转置走 `kutta_transpose`
  （`c = (integrals * (q / measure)[None, :]).sum(axis=1)`，再 `np.add.at` 到 upper/lower 面）。
- `rhs()` 只依赖源项：`source_apply(sigma)`，同样逐块收缩，不形成源矩阵。
- `dense()` 只为审计存在（`mf_test.py` 用它和存盘矩阵对拍），生产路径不调用。

### 1.2 近场分块预条件 P⁻¹ / P⁻ᵀ

`NearFieldPreconditioner(op, depth=3)`：按面元中心的最长轴二分（`_bisect`）得到 8 个空间块，
取 P = [[blockdiag(D), 0], [−T, M]]：D 是块内物面偶极子子矩阵（对角含 −1/2 迹），
T 是 Kutta 行、M 是环量质量矩阵。`solve(r)` 前向用 `sla.lu_solve`，`transpose=True` 时
用 `sla.lu_solve(fac, ..., trans=1)` 得 P⁻ᵀ。预条件内存 = Σ块大小²，n=4911 时 2 967 048 条目
（23.7 MB），是稠密矩阵 24 120 321 条目的 12.3%。

### 1.3 重启 GMRES 与隐式伴随

`gmres(matvec, b, restart=80, rtol=1e-10, maxiter=2000, precond=None, transpose=False)`：
左预条件、每轮用**真实残差** `‖b − A x‖/‖b‖` 判停（不是预条件残差），失败返回实际残差不让调用方静默接受。
伴随用同一个算子转置：`lam = gmres(op.rmatvec, dL/dx, transpose=True, precond=pre)`，
梯度 `dL/dα = lam · (∂b/∂α)`，其中 `∂b/∂α = source_apply(−normal·∂flow/∂α)`。

### 1.4 半模门控与 R A E 降阶

`mirror_gate(op, tol=1e-9, station_tol=1e-6, freestream=None)` 逐项判定：面元镜像配对是否双射、
是否对合、是否有面元需要自镜像（`self_panels_on_plane` 报贴在对称面上的面元、`centreline_straddling`
报跨对称面的面元，两者都改变未知量计数）、TE 边是否镜像到**同一 bank** 的 TE 边（XZ 反射 y→−y 保 z，
所以 upper→upper、lower→lower；逐项报 `te_edges_paired_upper/_lower`）、站位图是否自映射、
来流是否含侧滑（`|v_y|/|v| > 1e-12` 直接拒绝）。任何一项不满足都返回 `status="blocked"` 并列出原因。

> 2026-10-30 更正：旧判据把 `pair[op.upper]` 与**全部 lower 面**求交，等价于要求反射交换 z bank，
> 任何 XZ 对称网格都不满足 → 真对称网格会被误判 `blocked`。节点级审计 `panel/symmetry.py` 的
> `te_bank_gate` 一直是 bank 一致的写法，两者现已统一。



`HalfModelReduction(nf, ng, mirror, station_mirror)` 只在门控通过时使用：`expand` 把半模状态镜像成全模，
`restrict` 折叠全模残差，`apply = R A E`，`apply_transpose = Eᵀ Aᵀ Rᵀ`（镜像方向梯度不丢）。

## 2. 正确性（`runs/mf_test.py`，single_wing 484 面 + 39 站位 = 523 未知量）

| 检查 | 结果 |
| --- | --- |
| `dense()` vs `solve_fields` 存盘矩阵 | max abs **3.719e-15** |
| `rhs()` vs 存盘 rhs | **4.163e-17** |
| `matvec` vs 稠密乘 | **6.661e-16** |
| `rmatvec` vs 稠密转置乘 | **1.554e-15** |
| 伴随恒等式 `<Av,w>` vs `<v,Aᵀw>` | 相对 **1.4e-16** |
| 预条件转置恒等式 `<P⁻¹r,w>` vs `<r,P⁻ᵀw>` | 相对 **2.03e-15** |
| 稠密 LU 真残差 / 与存盘状态差 | 1.15e-15 / 2.6e-15 |

`gmres` 与 `pgmres` 的状态对存盘 LU 状态（6 个算例，rtol 1e-10）：

| 算例 | n | 迭代 | 真实残差 | 状态 max abs 差 | 相对 |
| --- | --- | --- | --- | --- | --- |
| single_wing | 523 | 29（pgmres） | 1.27e-11 | 2.11e-11 | 4.8e-11 |
| two_bodies_coarse | 651 | 26 | 1.54e-11 | 1.92e-11 | 3.67e-11 |
| forward_body | 735 | 26 | 1.94e-11 | 2.90e-11 | 4.45e-11 |
| two_bodies_fine | 1681 | 22 | 4.27e-11 | 4.66e-11 | 7.23e-11 |
| aft_1.0001_size_0.15 | 2015 | 26 | 3.17e-11 | 9.75e-11 | 9.32e-11 |
| panel_big_case | 4911 | 25 | 2.92e-11 | 1.20e-10 | 1.19e-10 |

未预条件 GMRES 在同一 rtol 下：523 → 251 次、651 → 222、735 → 278（Jacobi 无效，见 issue #3 报告），
预条件把迭代数压到 22–29。

隐式伴随（single_wing，环量升力通道 `CL_circ = 2 Σ γ_i Δy_i / S`）：

- 前向 31 次迭代、残差 1.13e-12；伴随 32 次迭代、残差 1.21e-12，冷 1.40 s / 热 1.37 s。
- `<lam, Ax> = <Aᵀlam, x> = 3.183249678083e-01`。
- `dCL_circ/dα`：矩阵无关伴随 **3.596739366554**，JAX 反向 AD **3.596739366554**（相对差 **2.963e-15**）；
  中心差分 h=1e-3 / 1e-4 / 1e-5 相对差 1.667e-7 / 1.666e-9 / 1.307e-11。

### 2.1 原始载荷对拍（`runs/mf_load_check.py`，`runs/mf_audit/loads.json`）

`runs/mf_loads.py` 用 NumPy 复刻 `panel/differentiable.py:115-146` 的两列无加权切平面拟合 → 速度 → Cp → 力/矩，
先验证「用存盘 mu 重算」能重现存盘 `panel_forces`，再用 GMRES 解出的 mu 与存盘 mu 分别算载荷比差。

| 算例 | n | 迭代 | CL（存盘） | dCL 绝对 | dCL 相对 | dCD | dCM 最大分量 | dCp 最大 | 逐面力 L2 相对 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_wing | 523 | 29 | 0.241534412 | −8.6e-07 | −3.6e-06 | −7.5e-08 | 3.1e-07 | 2.5e-05 | 1.5e-06 |
| two_bodies_coarse | 651 | 26 | 0.263212951 | −7.8e-07 | −2.9e-06 | −5.9e-08 | 2.2e-07 | 2.8e-05 | 1.1e-06 |
| forward_body | 735 | 26 | 0.269042919 | −1.2e-06 | −4.4e-06 | −1.0e-07 | 3.4e-07 | 2.4e-05 | 2.0e-06 |
| two_bodies_fine | 1681 | 23 | 0.312071645 | −1.4e-06 | −4.6e-06 | −1.0e-07 | 3.7e-07 | 2.3e-05 | 3.9e-06 |
| aft_1.0001_size_0.15 | 2015 | 26 | 0.319644898 | −1.8e-06 | −5.7e-06 | −1.4e-07 | 4.7e-07 | 1.2e-03 | 3.7e-06 |
| panel_big_case | 4911 | 25 | 0.350839160 | −5.6e-07 | −1.6e-06 | +3.4e-08 | 2.9e-07 | 7.0e-04 | 2.5e-06 |

注意两点，避免误读：

- 载荷差（1e-6 相对）比状态差（1e-10 相对）大 4–5 个数量级，原因是**切平面模板的病态**，不是 GMRES：
  两列最小二乘的行列式在 single_wing 上最小 1.903e-06（中位 8.533e-03），放大到 5.25e+05；
  panel_big_case 更差（最小 7.033e-10，放大 1.42e+09）。dmu ~ 2e-11 经 5e5 放大即得 dCp ~ 1e-5，与实测一致。
  这正是 PR #10 已记录的「仿射模板模型误差」，与线性求解器无关。
- 复算几何本身也只能到 ~5e-8（`load_prepared_run` 从 `fields.npz` 的缩放坐标乘以弦长再加原点重建，
  重建后的 center/normal/area 与存盘值差 5.5e-8 / 9.6e-8 / 1.5e-8），所以「用存盘 mu 复算 vs 存盘 panel_forces」
  的闭合地板是 7.3e-09（力，q=1 无量纲）。上表的 dCL 是在**同一套重建几何**上比的，不受这个地板影响。

## 3. 性能与内存（独立进程，`runs/mf_bench.py`）

| 算例 | n | dense 组装 冷/热 | LU 冷/热 | LU 回代 冷/热 | GMRES 冷/热 | PGMRES 冷/热 |
| --- | --- | --- | --- | --- | --- | --- |
| single_wing | 523 | 0.478 / 0.036 s | 0.030 / 0.030 s | 0.157 / 0.054 ms | 10.65 / 10.22 s（251 it） | 2.02 / 1.54 s（29 it） |
| two_bodies_coarse | 651 | 0.507 / 0.055 s | 0.037 / 0.037 s | 0.185 / 0.066 ms | 33.2 s（222 it） | 1.99 / 1.59 s（26 it） |
| forward_body | 735 | 0.525 / 0.066 s | 0.040 / 0.040 s | 0.315 / 0.091 ms | 44.5 s（278 it） | 2.10 / 1.56 s（26 it） |
| two_bodies_fine | 1681 | 0.740 / 0.259 s | 0.062 / 0.064 s | 0.546 / 0.316 ms | — | 6.35 / 5.90 s（22 it） |
| aft_1.0001_size_0.15 | 2015 | 0.904 / 0.390 s | 0.069 / 0.070 s | 0.736 / 0.549 ms | — | 10.85 / 10.36 s（26 it） |
| panel_big_case | 4911 | 2.089 / 1.428 s | 0.256 / 0.233 s | 4.65 / 4.50 ms | — | 40.4 / 39.5 s（25 it） |

| 算例 | n | dense 峰值 RSS | PGMRES 峰值 RSS | dense 工作集 起→终 | PGMRES 工作集 起→终 |
| --- | --- | --- | --- | --- | --- |
| single_wing | 523 | 339 MB | 355 MB | 172 → 302 MB | 172 → 269 MB |
| two_bodies_coarse | 651 | 348 MB | 344 MB | 171 → 311 MB | 172 → 262 MB |
| forward_body | 735 | 350 MB | 352 MB | 172 → 315 MB | 171 → 269 MB |
| two_bodies_fine | 1681 | 538 MB | 499 MB | 173 → 365 MB | 173 → 284 MB |
| aft_1.0001_size_0.15 | 2015 | 631 MB | 608 MB | 174 → 385 MB | 174 → 292 MB |
| panel_big_case | 4911 | **2178 MB** | **1817 MB** | 177 → **708 MB** | 177 → **298 MB** |

内存账很清楚：dense 路径常驻 A 和 LU 因子两份 n²（n=4911 各 193 MB），矩阵无关只额外留预条件
（23.7 MB）。n≤2000 时两者峰值只差 2–40 MB，因为 JAX/XLA 运行时本身占 170–300 MB，
把几 MB 的矩阵淹没了；n=4911 才开始分离（工作集 531 MB vs 121 MB）。

外推到 23536 面（仅作容量估算，未运行）：dense 需 2×23536²×8 B = 8.86 GB，
矩阵无关需 ≈ 0.125×23536²×8 B = 0.55 GB 预条件 + O(n) 向量，可以进 8 GB 显存的卡。
但单次 matvec 仍是 O(N²)，这是本实现的硬边界——它不是 FMM。

## 4. 半模降阶

### 4.1 在严格镜像系统上验证（VortexAD 1616 面 AIC）

`runs/mf_half_vortexad.py` 使用 `runs/backend_probes/aic_symmetry_matrices.npz`（body/wake/source/full/rhs/mirror）：

- 配对：1616 面，unpaired 0，对合 True，半模 808 未知量（50%），条目 652 864 vs 2 611 456。
- `R A E` 与 `A_RR + A_RL` 差 **0.0**（逐位相同）。
- 半模解 vs 全机解相对 **6.409e-04**（max 1.251e-02）——与 issue #4/PR #8 已定位的 wake/Kutta 块不对称一致，
  降阶本身没错，是后端 AIC 组装不对称。
- 残差：全机 1.37e-15、半模 7.70e-16、镜像重建全机 5.47e-06；降阶伴随恒等式相对 1.97e-16。

### 4.2 Dpanel 自己的网格：全部 blocked

| 算例 | 面元 | 有精确镜像伙伴 | 最近镜像距离中位 | 站位配对 | 门控 |
| --- | --- | --- | --- | --- | --- |
| single_wing | 484 | 26 | 1.329e-2 | 41/41 对合 | blocked |
| two_bodies_coarse | 616 | 56 | 1.646e-2 | 45/45 对合 | blocked |
| forward_body | 696 | 74 | 1.574e-2 | 41/41 对合 | blocked |
| two_bodies_fine | 1646 | 68 | — | 41/41 对合 | blocked |
| aft_1.0001_size_0.15 | 1976 | 148 | — | 41/41 对合 | blocked |

原因统一：Gmsh 的三角化在左右两侧不镜像（站位对称、三角化不对称），484 面里还有 7 个中心线面元
需要自镜像、64/80 条 TE 边镜像不到已有下 TE 边。按契约这些算例的半模列**标 blocked**，
没有改网格去冒充同离散等价（那会改变未知量数与离散误差）。

侧滑门控实测：干净来流 `sideslip_ratio=0.0`；来流 `[1, 0.05, 0]` 时 `sideslip_ratio=4.994e-02`，
理由写入 `"freestream carries a sideslip component (|v_y|/|v| = 4.994e-02); the mirror symmetry is broken"`。

## 5. 判读

1. **矩阵无关算子是正确的**：与稠密矩阵逐元素 1e-15 级一致，伴随与预条件转置都满足恒等式，
   状态与 LU 解在 rtol 下一致到 1e-10，伴随梯度与 AD 一致到 2.96e-15。
2. **内存收益只在内存真正吃紧时出现**。n≤2000 峰值只降 2–40 MB（运行时主导），n=4911 时
   dense 2178 MB vs pgmres 1817 MB、工作集 708 vs 298 MB；外推 23536 面 dense 8.86 GB 不可行、
   矩阵无关约 0.6 GB 可行。
3. **速度上是倒退**：预条件 GMRES 比 LU 慢 3.4×（523 面：1.54 s vs 0.030 s + 0.054 ms 回代），
   4911 面慢约 6.5×。原因是单次 matvec 仍 O(N²)（4911 面 1.6 s/次，LU 回代 4.5 ms）。
   所以本任务只声称「省内存」，不声称「加速」。
4. **半模目前对 Dpanel 不可用**：降阶机制在严格镜像系统上验证通过（RAE 逐位等于 A_RR+A_RL），
   但 Gmsh 三角化不镜像，五个算例全 blocked。要用半模必须先有半边建模+镜像的网格生成路径，
   而不是事后对称化 AIC（那会掩盖 6.4e-4 的组装偏差）。
5. **不是 FMM**：没有多极/分层压缩，单次作用 O(N²)；要真正跑到 23536 面还需要低秩或多极。
6. 预条件内存仍是 O(n²/8)（8 块，每块 n/8），n=23536 时约 0.55 GB；块数/深度可调，
   更深的树会进一步降内存但削弱预条件质量。

## 6. 复现

```powershell
$env:JAX_PLATFORM_NAME="cpu"; $env:TMP=$env:TEMP=$env:TMPDIR="D:\3D\Projects\panel\runs\pipwork"
.\\.conda_env\\python.exe runs\\mf_test.py single_wing        # 一致性审计
.\\.conda_env\\python.exe runs\\mf_bench.py single_wing pgmres # 独立进程基准
.\\.conda_env\\python.exe runs\\mf_adjoint.py single_wing       # 隐式伴随 vs AD vs FD
.\\.conda_env\\python.exe runs\\mf_load_check.py               # 载荷对拍
.\\.conda_env\\python.exe runs\\mf_half_vortexad.py            # 半模降阶验证
```