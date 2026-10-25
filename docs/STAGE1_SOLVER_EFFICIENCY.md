# Dpanel 第一阶段：LU / GMRES 与镜像半模验收计划

状态：算法和代码审查 + 已有本地回执；本 PR 的新实验**尚未执行**。

## 已有证据

1. VortexAD 的非结构化 PanelMethod 读取全部 cell，并以完整单元集合构造 AIC_mu。当前 810 节点、1616 三角面元测试没有启用机体 XZ 半模降阶。
2. FLOWPanel 的镜像网格工具将半网格变为完整网格；Cessna 的求解使用完整 Airframe（23536 面元）。FLOWPanel 另有地面效应镜像涡，与机体 y=0 对称降阶不同。
3. VortexAD 1616 面元本地 CPU 热前向约 0.55 秒，完整 alpha+FD pytest 冷启动约 159–166 秒；局部面力对单网格节点坐标的导数与中心差分相对误差约 4.9e-8。以上没有证明大批量整机速度。
4. VortexAD 默认 Cp 截断 -5 已实测改变 26/1616 面元，使近场 CDi 变化 -26.7%；未裁剪 Cp 最小值为 -274.33。关截断不能代替独立网格和局部压力精度验证。
5. FLOWPanel 23536 面元 Cessna 的 Float64 五个同阶稠密块估算为 22.16 GB = 20.64 GiB；本机发生内存不足。Float32 求解虽退出但物理量严重失效，不可作为有效性能基线。1440 面元开体直接法与水密体 LS 的气动结果不同，不能因为 LS 较快便视为等价。

## 算法判断

- Dense LU 分解 O(N^3)，保存 A 为 O(N^2)，同矩阵多右端回代 O(N^2)。
- 用密集 A 的 GMRES 每次矩阵作用 O(N^2)，k 次约 O(kN^2)，另有 Krylov 正交化与预条件器成本；并不保证 k 小于 N，也不保证比 LU 快。
- 用 LinearOperator 包装已保存的 A 只是无显式求逆，不是 Matrix-free 低内存面元法。
- 不保存完整 A 的 Matrix-free GMRES 仍可能每次需要 O(N^2) 的面元相互作用；FMM/分层远场才可能使单次作用接近线性或准线性，需另外处理几何 VJP 和转置伴随。
- 对固定线性问题 A.mu=b，隐式导数需额外求 A.T.lambda=g；优化不能只测前向。

## 已提交的可复现探针

第一项：test/export_vortexad_aic.py
- 用当前 VortexAD 面元图导出原始 AIC_mu 和 RHS 到 runs/backend_probes/vortexad_aic.npz。
- 记录 CSDL 构图、JAX 编译/第一次评估耗时，不要求保存历史 Cp，默认拒绝超过 3000 未知量的稠密矩阵。
- 只是导出真实离散矩阵，不是实现 Matrix-free。

第二项：test/probe_linear_solvers.py
- 读相同 A 和 b，在 float64 上对照 SciPy LU、无预条件 GMRES、诊断性 Jacobi 预条件 GMRES，以及 A.T 的转置伴随方程。
- 输出矩阵字节、LU 分解/回代时间、GMRES 迭代与 matvec 次数、原方程相对残差、与 LU 解差异。若预条件器失败要报告，不得宣布数值收敛。

第三项：test/test_probe_linear_solvers.py
- 自测上述测量工具的前向和转置 GMRES 路径，不等于 VortexAD 已验收。

## 本地执行

在项目内隔离的 VortexAD/CSDL/JAX 环境，先运行：
    python -m pytest -q test/test_probe_linear_solvers.py

再按内存预算运行：
    python test/export_vortexad_aic.py --out runs/backend_probes/vortexad_aic.npz --max-n 3000
    python test/probe_linear_solvers.py runs/backend_probes/vortexad_aic.npz --rtol 1e-8 --restart 40 --maxiter 20 --out runs/backend_probes/solver_comparison.json

准确记录环境 SHA、命令、退出码、阶段耗时、实际峰值 RSS、预条件/无预条件的相对残差与迭代数。A.nbytes 不等于完整进程峰值内存。首次/热启动、BLAS 线程、CPU/GPU、矩阵维数都必须记录。

## 后续对称半模（Issue #4）

对于 y=0 对称飞机、对称飞行状态，先比较显式全模的四个块与两半对称映射，再验证半模 A_half=A_RR+A_RL、右端、Kutta/尾迹、压力积分、总力、力矩和伴随梯度。外法向与双层势镜像符号要由真实公式和基准验证。半模不应添加 y=0 上虚假的物面封盖。

若半模未知量比完整模型减少一半，理想稠密 LU 分解约降至 1/8、单矩阵存储降至 1/4；这不是整体运行时间保证，也不意味着 VortexAD/FLOWPanel 当前有此特性。非对称飞行状态不能强行使用半模。

## 优先级

P0：先验收未裁剪 Cp 的局部精度和物理载荷。
P1：用真实 AIC 测量构造/求解/转置伴随，判断是组装还是求解瓶颈。
P2：在完整等价性测试后才实现机体半模。
P3：若内存与组装为主瓶颈，进入 Matrix-free、FMM、分层低秩与可微转置算子；神经粗空间和 Transolver 预条件属于第二阶段研究。

不修改 Dpanel 的 aero.py 默认模式，不启动生成器训练，不将测试脚本提交等同于验收。

## 本地真实 AIC 求解器测量（2026-10-25，PR #5）

来源：`https://github.com/Stevenkerman27/Dpanel/pull/5#issuecomment-6393503374`，本地锁定 VortexAD `ab64939`、CSDL-alpha `d4bf50e`、JAX 0.10.2 CPU、SciPy 1.17.1、OpenBLAS 0.3.31.188.0。**以下时间是已存稠密 AIC 上的 LU/GMRES，不包含影响矩阵生成与压强后处理。**

| N | LU 分解 s | LU 正向回代 s | GMRES 正向 s / 迭代 | GMRES 转置 s / 迭代 |
|---:|---:|---:|---:|---:|
| 1616 | 0.0596 | 0.000602 | 0.0236 / 91 | 0.0382 / 135 |
| 1712 | 0.0604 | 0.000590 | 0.0221 / 88 | 0.0398 / 129 |
| 2848 | 0.0923 | 0.001480 | 0.0587 / 79 | 0.0945 / 117 |

求解容差 `rtol=1e-8`，GMRES restart=40、maxiter=20 周期；已报告的正向及转置 `info=0`，真相对残差约 `1e-8`。1616 规模的 LU 正向及转置残差在 `1e-15` 量级。Jacobi 对角预条件基本没有节省迭代数（例如 91→89），不应称为有效优化。

对一个固定矩阵、多个右端，LU 分解后每次回代仅约 0.4–1.5 ms，因此在这些具体样本上**约 2–3 个右端便可能比单次次次 GMRES 更划算**；正向和转置伴随可复用相同 LU 因子。该结果不代表所有规模上 LU 更快，因为整体内存、面元几何变化和构造成本仍起关键作用。

实测图构建仅 0.107–0.184 秒，JAX 编译与首次 AIC 评估 **混合**耗时约 1.035–1.932 秒；尚无独立 AIC 装配纯耗时。禁止把导出脚本的这个值直接当成高频热运行组装成本，也禁止将已有 NPZ 的 GMRES 时间描述为训练总时间。

### 半模进一步诊断（Issue #4）

来源：`https://github.com/Stevenkerman27/Dpanel/issues/4#issuecomment-6393719897` 及后续评论，1616 面元 NACA0012 分成 808 + 808；相同后端的强度镜像相对差 `1.9e-4`，人为在测试中按镜像平均 AIC 后下降为 `6.28e-15`。但未经修复的 AIC 镜像条目最大相对差约 `1.66e-5`、对角差最高 `6.76e-6`；几何重心镜像误差低于 `9e-16`。**对称化验证只是定位问题，并不是认可在生产中修补 AIC**。

下一个独立诊断分支 `research/vortexad-aic-symmetry-diagnostics` 将拆分比较未加尾迹的本体双层 AIC、尾迹/Kutta 修正、源影响和 RHS，定位源/尾迹/自影响误差。该分支未运行就不得声称修复了对称性或通过升力载荷/梯度验收。
