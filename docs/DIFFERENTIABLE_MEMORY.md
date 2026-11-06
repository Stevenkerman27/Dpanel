# 可微流程的反传内存：外部做法与本仓库诊断

更新：2026-11-07。范围：P1 固定尾迹（`panel/p1_cst_gradient.py` → `tools/web/optimize_runner.py`）的导数链路。
状态：**已实现并实测**——1720 面板（size 0.1 m）单次评估峰值 15,273 MB → 1,810 MB；系数与梯度与改前一致（相对差 ~7e-9）。数字都来自本机实测脚本，路径随数字给出。

## 一句话

15 GB 不在前向、不在 Hessian、也不是泄漏，而是**装配的 backward 一次性重算了全部源块**：
`doublet_matrix` 被整段包在一个 `torch.utils.checkpoint` 里，反向时 27 个块的中间量同时物化。
把 checkpoint 下沉到**每个源块**，峰值立刻降到 1.7 GB。隐式节点（自定义 `autograd.Function`）是必需的第二步——
没有它反向要穿过 `create_graph=True` 的二次图，但**只做它并不会降内存**（实测仍然 15,273 MB）。

## 3. 机制

先纠正文档早前版本的判断：二次图**不是**主因。把隐式链改写成自定义 `autograd.Function`（反向只做一次一阶反传、
不再 `create_graph=True`）之后，峰值仍然是 15,273 MB。逐项隔离才定位到真正的原因（`runs/scratch_opt/mem_asm.py`，
1720 面，只对装配求一次梯度）：

| 只反传 | 峰值 |
| --- | --- |
| `A`（壁面 doublet） | 15,272 MB |
| `grad` + `W` + 面积 | 2,379 MB |
| `B`（含 `solve_triangular`） | 15,280 MB |

`panel/p1_assembly_torch.py` 的 `doublet_matrix` 内部本来就有 `face_chunk` 循环，但
`galerkin_operators` 的 `_recompute` 把**整个调用**包成一个 checkpoint。反向时 checkpoint 重算这个函数，
27 个块的图在重算过程中**全部建立**，每个块约 0.4–0.6 GB，合计 15 GB。这解释了两个反直觉现象：

- `face_chunk=8` 与 `face_chunk=64` 的峰值几乎相同（15,461 vs 15,337 MB）——块大小只决定「多少块」，不决定「同时活多少」；
- `_recompute` 打开只比关掉省 1.4 GB（16.9 → 15.5 GB）——因为省下的只是前向保存的少量张量，反向新建的一分没少。

## 4. 实现（已落地）

### A 隐式节点：`_FixedWakeImplicit`（`panel/p1_cst_gradient.py`）

`torch.autograd.Function`，输入是 cst 几何（`xyz`/尾迹/参考面积），输出 (CL, CD[, CDi], info)。

- forward：`no_grad` 下装配 → Kutta 求解 → 算系数，只存 `xyz, wake, reference_area, z, reduction, hessian`；
- backward：在 `torch.enable_grad()` 内重建一次装配，先解 `H^T λ = ∂J/∂z`，再用
  `total = (coef_x · grad_coef).sum() − (λ · residual_x).sum()` 对几何做**一次** `autograd.grad`。
  这样 `∂g/∂x` 只需要一次 vjp，不再有 autograd-through-autograd。
- 关键推导：`F = reduction^T (B^T (B s − rhs) + 25 P(s)^T p(s))` 对 `(B, rhs, grad)` 是显式代数式，
  `λ^T F` 对几何是一阶函数，所以整条反向链只有一阶。
- 坑：`backward` 默认在 `no_grad` 下执行（必须显式 `enable_grad`）；`result.reduced` 的列数由
  `result.reduction` 决定（86→77），必须用 `reduction` 而不是自己算的 `linear_basis` 去缩并。

### D 逐块 checkpoint（`panel/p1_assembly_torch.py`）——真正降内存的一步

把块内计算抽成 `_doublet_block(t, X, mask)`，每次调用用 `_checkpointed`（`use_reentrant=False`，仅在需要梯度时启用）包一层。
于是反向是「一个块重算 → 立刻反传 → 释放」的循环，峰值 = 单块中间量 + 梯度缓冲。

### C 正交项（未做，留作 0.05 m 的后续）

1. 梯度路径用 3 点求积替代 `_duffy_rule(order=3)` 的 12 点：装配规模 4× 下降，需先用
   `test/test_p1_cst_gradient.py` 的系数/梯度比对确认误差可接受。
2. 矩阵无关（GMRES + 伴随），不存 (20640, 903) 的 A。
3. float32 只用于伴随——不推荐，隐式 Hessian 条件数对精度敏感。

## 5. 实测验收

脚本：`runs/scratch_opt/mem_implicit.py`（完整 implicit 路径）、`runs/scratch_opt/mem_asm.py`（只装配）、
`runs/scratch_opt/implicit_cmp.py`（与改前基线 `runs/scratch_opt/opt02/implicit_ref.json` 逐项比对）。
内存用 PowerShell `Start-Process` + 轮询 `WorkingSet64` 采样（峰值，不是平均）。

| 场景 | 改前 | 改后 |
| --- | --- | --- |
| 1720 面，只反传 A | 15,272 MB | 1,697 MB |
| 1720 面，完整 implicit（含一次反向） | 15,273 MB | 1,810 MB（`recompute` 默认）/ 1,840 MB（关掉） |
| 1720 面，前向 / 反向耗时 | 7.3 s / 13.4 s | 7.3 s / 17.9–20.5 s |
| 3400 面（size 0.05） | 30 GB 级，eval 1 之前就死 | 3,848 MB，能跑完前向+反向 |

正确性（同一 `alpha=4`、同一网格）：

- size 0.5（132 面）：CL/CD/CDi 12 位小数与改前**完全一致**，`dJ/darea = −9.834226837932e−02` 一致，cst 梯度最大相对差 6.91e−09；
- size 0.35（156 面）：最大相对差 5.00e−09，`dJ/darea = −9.856343341726e−02`；
- 差值来源是 BLAS 求和顺序，不是公式改动。
- `pytest test/test_web_grass.py test/test_web_grass_geometry.py test/test_web_solve_threads.py test/test_web_step_leading_edge.py` → 16 passed；
  `test/test_p1_cst_gradient.py test/test_p1_surface_gradient.py` → 11 passed。

**没达到的验收项**：单次反向时间没有变快（13.4 s → 17.9 s），因为反向现在要多做一次装配重算。这是拿时间换内存的自觉选择；
若要两全，需要 C.1（求积点）或 C.2（矩阵无关）。

**0.05 m 的遗留问题**：内存已经不是瓶颈（3.85 GB），但 3400 面上 pressure-Kutta 不收敛——
`status=3`、`stationarity_inf=3.28e−07`（`max_nfev=160`）/ `9.58e−08`（`max_nfev=400`），被 `fixed_wake_geometry` 的
stationarity 门禁拒绝。这是数值精度上限，与内存无关，不要在优化器里绕过它。

仍然 `physical_accuracy_accepted=False`：本文是内存与导数链的验收，不是气动精度验收。
