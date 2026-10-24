# 固定网格 node/weak Kutta 同批比较

协议先于本次计算冻结。运行脚本为 [run_kutta_checks.py](run_kutta_checks.py)，固定输入为 `results/strips_sequence_v1/s24/mesh/mesh.npz`，网格不重建。结果写入 `results/kutta_checks_v1/`；若目录已存在，脚本拒绝覆盖。所有工况请求 NCPU=8。

## 算例和预设门禁

在完全相同的 s24 网格上，用同一个 `solve.py` 版本运行以下七例：

| Kutta 形式 | Mach | α |
|---|---:|---:|
| node | 0 | −5°, 0°, +5° |
| weak | 0 | −5°, 0°, +5° |
| weak | 0.2 | +5° |

零攻角检查对 node/weak 各自分别要求 `|CL_pressure|`、`|CL_jump|` 和 `max|Γ| < 1e-9`。Mach=0 的 ±5° 各自检查 `|CL_pressure(+5)+CL_pressure(−5)|`、对应 `CL_jump` 之和以及展向离散 `max|Γ(+5)+Γ(−5)| < 1e-9`。每例要求最终一次记录的 nonlinear `residual_norm` 与 `kutta_max` 均有限且绝对值 `<1e-9`；全 history 中所有数值字段均须有限；所有实际记录的 `linear_relative_residual` 均须有限且绝对值 `<1e-9`。初始及中间 nonlinear residual 允许大于该门限，不作为失败条件；它们只要求有限，求解过程保持原样，不作修改。另检查命令退出码、收敛状态和 Cp 有限性。这些门禁只验证内部数值/对称性，不证明气动准确度。

不对 CL、CD、Cp、壁面速度或 wake 量拟合参考值，也不以 node 结果作为 weak 的物理真值。Mach 0.2 weak 工况只作独立记录，不混入 Mach=0 的零攻角/反对称门禁。

## 保存数据

每例保存完整 solver `result.json`、`field.npz`、合并 stdout/stderr 的 `run.log`、命令与退出码、wall time、所有载荷/线性和非线性残差历史、Cp 元素/采样数组 finite 数及极值、全局和分区壁面法向速度统计、各 tag 原始质量通量、wake 原始配对质量通量跳跃/压力跳跃/法向速度诊断，以及该例对应的 `solve.py`、`p2.py` 和 `PROTOCOL.md` 源快照。原始 Cp 数组不裁剪，直接留在 solver 输出的 `field.npz`。summary 另保存输入 mesh 与 metadata、solver/P2/protocol/executable hashes，并检查七例实际实现版本和 mesh hash 一致。

最初脚本版本错误地把 history 中所有 nonlinear residual（包括正常较大的初始未收敛残差）都纳入 `<1e-9` 门禁。七例已完成的原始求解与迭代未改动；修订脚本使用 `--reuse-existing` 从已保存 JSON/NPZ 重算验收字段，不重跑 solver。初始门禁摘要另存为 `summary_pre_gate_correction.json` 以保留判定修订轨迹。

既有 `weak_kutta_v1/s24_a5` 与 `s48_a5` 只作旁证。它们各有保存的源快照且来源版本不同于本次可能运行版本；不能把其载荷与本次 node 或 weak 结果拼成同批对照。本次所有 node/weak 工况统一现场运行，形成独立配对证据。

## 解释限制

此处是固定网格上的 Kutta 约束离散形式对照，不是网格收敛、物理解认证或外部实验比较。弱式 Kutta 使用主 solver 实现的同一原始顺弦速度差，对 Gamma 展向 hat 测试函数积分，并按 station 区间切分后缘边；本脚本不另建 Kutta 定义。应分别陈述“弱质量守恒残差”与“raw flux/wake诊断”，不可把弱式方程守恒误称为逐面原始通量连续。压力阻力可含离散误差与尾迹模型影响，不把 `CD_pressure` 直接等同升力线 `CDi`。

新批次运行方式：`python lifting3d/run_kutta_checks.py`。仅重算本批次既有结果门禁：`python lifting3d/run_kutta_checks.py --reuse-existing`。脚本只调用现有 `solve.py` 读取指定的固定网格，不调用网格 builder。

## 实际结果

七例首次运行的命令退出码均为 0，solver 均报告 `converged_subsonic_unvalidated_lifting`。复核摘要为 `results/kutta_checks_v1/summary.json`；首轮错误门禁的历史摘要保存在 `summary_pre_gate_correction.json`。七例 mesh、solver、P2、solver protocol hashes 分别完全一致：

- Mesh: `c26bf3037a9c6857a49596e2d9eb75186a54b30bad6ade2e86813bc357fff193`
- Solver: `7d344c5e4530872c9d714a8619491a49d3becb54148ddb01921b4de73adad04a`
- P2: `0047de5eb0a63768d97037e7a69b18defbc2a7da726044999e8aeba449da548f`
- `PROTOCOL.md`: `b03bc75d1b0dfb0376331d46e0c9dc5515481cdece992777d30a280294aa186a`

每个 case 子目录都保存其实际 solver、P2、protocol 快照，三个快照的 hash 与该例 `result.json` 中的对应字段匹配。当前 `solve.py` 和 `p2.py` hash 仍与运行时一致；当前根目录 `PROTOCOL.md` hash 为 `db2775feaaba442abdd174ffc215bb5aea768cca42230aadc8c44dff22cb92da`，不同于七例实际使用的 protocol 快照 `b03bc75d...`，表明该文档在本批次后有改动。summary 将运行时快照和当前文件 hash 分字段记录；对本批结果的解释以逐例保存的运行时 snapshot 为准。

载荷、Cp、全局壁面和 wake 汇总：

| case | CL_pressure | CL_jump | CD_pressure | Cp min / max | wall normal RMS / max | wake raw mass-flux jump RMS / max | wake Cp jump RMS | wake upper normal RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| node, M0, −5° | −0.333212267 | −0.337037817 | 0.009650345 | −1.729917 / 0.999851 | 0.029014 / 1.482264 | 0.008375 / 0.152879 | 0.002151 | 0.047421 |
| node, M0, 0° | −3.97e-15 | −2.25e-16 | 0.001185633 | −1.326574 / 0.999795 | 0.017808 / 1.037232 | 0.008407 / 0.153463 | 4.13e-15 | 0.004204 |
| node, M0, +5° | 0.333212267 | 0.337037817 | 0.009650345 | −1.729917 / 0.999851 | 0.029014 / 1.482264 | 0.008375 / 0.152879 | 0.002151 | 0.047166 |
| weak, M0, −5° | −0.340129402 | −0.344038162 | 0.010095309 | −1.733469 / 0.999871 | 0.029462 / 1.484091 | 0.008375 / 0.152879 | 0.002714 | 0.049373 |
| weak, M0, 0° | −4.47e-15 | −1.05e-15 | 0.001185633 | −1.326574 / 0.999795 | 0.017808 / 1.037232 | 0.008407 / 0.153463 | 4.32e-15 | 0.004204 |
| weak, M0, +5° | 0.340129402 | 0.344038162 | 0.010095309 | −1.733469 / 0.999871 | 0.029462 / 1.484091 | 0.008375 / 0.152879 | 0.002714 | 0.049235 |
| weak, M0.2, +5° | 0.344540581 | 0.348507769 | 0.010314318 | −1.781152 / 1.009875 | 0.029510 / 1.460755 | 0.008360 / 0.153129 | 0.002760 | 0.049215 |

每例 raw Cp 共 `149,904` 个样本，finite `149,904/149,904`，nonfinite 为 0；未裁剪，完整数组保存在对应 case 的 `solve/field.npz`，`summary.json` 记录数组路径与 SHA256。每例的全局 raw mass flux by boundary tag，以及 interior/leading/trailing/tip 四个壁面区域各自的面积、normal RMS/max、Cp min/max，均完整保存在总 summary 和逐例 solver result JSON。wake raw mass-flux jump 和压力跳跃是成对上/下 wake 面的独立诊断，不代表跨整个内部网格面做了 flux audit。

修正后的数值门禁结果：

| Kutta 形式 | 零攻角最大误差 (`CLp`, `CLjump`, `max|Γ|`) | ±5° 最大反对称误差 (`CLp`, `CLjump`, `max|Γ+ + Γ−|`) | 最大最终 nonlinear residual | 最大最终 Kutta | 最大已记录 linear relative residual | 结果 |
|---|---:|---:|---:|---:|---:|---|
| node | 3.97e-15 | 3.75e-15 | 7.70e-14 | 2.49e-14 | 4.21e-13 | 通过 |
| weak | 4.47e-15 | 4.88e-15 | 8.84e-14 | 3.42e-14 | 5.65e-13 | 通过 |

七例中最大的初始 nonlinear residual 为 `0.02424`，零攻角初值为 `0.01724`；这是 history 的起始迭代状态，不作为门禁失败。Mach 0.2 weak 解经过中间 residual `3.32e-4`、`5.30e-8` 后收敛到 `8.30e-14`，solver 迭代未因门禁修订而改变。七例已记录的线性残差均有限且小于 `5.65e-13`；每例完整 history 见 summary。

这次固定网格比较显示 weak 相对 node 在 Mach=0、α=5° 时 `CL_pressure` 从 `0.333212` 变为 `0.340129`、`CL_jump` 从 `0.337038` 变为 `0.344038`；这是不同 Kutta 离散下的数值差异，不足以单独判定哪种载荷更接近物理真值。两种形式的 raw wake flux jump 仍约为 RMS `8.38e-3`、max `1.53e-1`，弱式质量守恒残差约 `5.8e-14` 至 `6.0e-14`；弱式代数守恒不能解释为 raw flux 连续。固定尾迹、当前网格和有限厚度三维气动准确性仍未由此测试认证。
