# PR #8 本地实测：VortexAD 镜像 AIC 不对称的来源分离

日期 2026-10-25。执行 PR #8（head `7d18b3a`，base `stage1/linear-solver-symmetry-audit`）的复现契约：
只做诊断，不改上游矩阵、不重跑 Cessna、不把对称化当生产修复。

## 1. 环境与命令

- 环境：`.conda_env` Python 3.11.17 / numpy 2.4.6 / scipy 1.17.1 / jax 0.10.2 (CPU) / csdl_alpha 0.1.1；
  VortexAD 源码 checkout `runs/backend_probes/src/VortexAD`（含 sample mesh），`PYTHONPATH` 指向它，`JAX_PLATFORM_NAME=cpu`。
- 网格：`naca0012_LE_TE_cluster.stl`，n = 1616 面元；`alpha = 5°`、`Mach 0.2`、`ref_area 10`、`Cp cutoff -5`（不影响 AIC/RHS）。
- 分支检出：`runs/pushrepo` 的 worktree `runs/pr8_work`（detached `7d18b3a`）。

`python -m pytest -q test/test_probe_vortexad_aic_symmetry.py` → **2 passed**, 0.29 s, exit 0
（只有 pytest cache 目录的 WinError 5 警告，属沙箱限制，不影响断言）。

`python test/probe_vortexad_aic_symmetry.py --out runs/backend_probes/aic_symmetry.json` → exit 0，
JAX 编译+首跑 1.009 s。产物：`runs/backend_probes/aic_symmetry.json`、`aic_symmetry.log`、
`aic_symmetry_matrices.npz`（本地附加导出，便于离线做逐块归因）。

## 2. 镜像配对

| 项 | 值 |
|---|---|
| 面元数 / 半模数 | 1616 / 808 |
| 重心镜像最大距离 | **0.0** |
| 法向镜像最大偏差 | 5e-12 |
| 双射且对合 | 通过（未抛异常） |
| 尺度 | 10.0863 |

配对是用几何做的（KD-tree + 反射 + 对合检查），不依赖 STL 三角面顺序，与 issue #4 的结论一致。

## 3. 四部分矩阵与 RHS 的镜像差

| 块 | max abs Δ | 相对全局峰值 | RMS Δ | 行最大值中位数 | 对角 max Δ | >1e-12 条目 | 该块峰值 |
|---|---|---|---|---|---|---|---|
| `AIC_mu_orig`（body doublet） | 2.240e-06 | 4.48e-06 | 2.88e-09 | 1.58e-14 | 2.240e-06 | 790 | 0.5000 |
| `AIC_mu - AIC_mu_orig`（wake/Kutta 调整） | **8.309e-06** | **2.42e-04** | 4.12e-08 | 1.76e-07 | 6.755e-06 | 96748 | 0.0344 |
| `AIC_sigma`（body source） | 3.383e-12 | 7.53e-11 | 6.41e-15 | 2.93e-16 | 2.240e-12 | 16 | 0.0450 |
| `AIC_mu`（完整） | 8.309e-06 | 1.66e-05 | 4.13e-08 | 1.76e-07 | 6.755e-06 | 97508 | 0.5000 |
| `RHS` | 8.855e-11 | 2.33e-11 | — | — | — | — | — |

结构：

- **body source 严格对称**（3.4e-12，相对 7.5e-11）→ 核函数/三角绕序没有问题。
- **body doublet 只有对角不对称**：808 个对角里有 312 个 >1e-12，off-diagonal 最大只有 1.93e-08。
  即自影响项 `-0.5` 在镜像面元上算出了 2.24e-06 的绝对差（相对 4.5e-06）。
- **wake/Kutta 调整是主要来源**：8.31e-06 绝对差、相对该块峰值 2.4e-04，扩散在 96748 个非零条目上，
  行最大值 >1e-6 的有 142 行、>1e-7 的有 1180 行（共 1616 行）。
- 最坏的几行都在后缘：`x=0.9897, y=±5.0010`（翼梢后缘，Δ=8.31e-06）、`x=0.9897, y=∓5.0010`（下表面，7.68e-06）、
  以及 `x=0.9948` 的一排展向站位（6.7e-06）。
- `RHS` 对称到 8.9e-11（相对 2.3e-11）。

## 4. 归因：半模降阶误差到底来自哪一块

用同一批矩阵做半模降阶实验（`A_half = A_RR + A_RL`，`b_half = b_R`，解出后镜像回填比全机解，度量相对 L2 误差），
逐块对称化（`A_sym = 0.5(A + A[m(i),m(j)])`）看误差怎么变：

| 变体 | 半模 vs 全机相对误差 | max abs | 全机残差 |
|---|---|---|---|
| 原始（as is） | 6.41e-04 | 1.251e-02 | 1.37e-15 |
| 只对称化 body doublet | 6.41e-04 | 1.251e-02 | 1.49e-15 |
| **只对称化 wake/Kutta** | **4.75e-08** | 9.692e-06 | 1.44e-15 |
| 只对称化 body source | 6.41e-04 | 1.251e-02 | 1.39e-15 |
| 两者都对称化 | 8.07e-13 | 1.576e-10 | 1.47e-15 |

结论明确：**半模降阶的 6.4e-04 解误差 100% 来自 wake/Kutta 调整块**；
body doublet 的对角不对称（2.2e-06）与 body source 的不对称（3.4e-12）对解级等价性没有可测贡献。
只修 wake 块就能把半模误差压到 5e-08；两者都修就是机器精度。

注：`aic_symmetry_matrices.npz` 里的 `full` 与之前 `vortexad_aic.npz` 的 `A` **逐位相同**（max abs diff 0.0），
所以这次归因和 issue #4 的 1.9e-04 是同一套数据（那里用的是另一种归一化度量，量级一致）。

## 5. 给上游的判读

PR 描述里的两个假设都排除了：不是核/三角绕序（source 与 body doublet 的非对角都对称到 1e-8/1e-11），
也不是 Kutta 条件本身，而是 **wake 影响系数（Kutta 调整项）的组装**——它把本该镜像相等的系数算出了
相对自身 2.4e-04 的偏差，并在半模降阶里被条件数 662 放大成 6.4e-04 的解误差。

建议的下一步（不在本次改动范围）：

1. 查 Kutta 调整项的装配：镜像尾迹面元是按反射生成还是按展向 strip 顺序生成的；
   尾迹双强（`mu_wake`）的符号/归一化是否带 y 相关因子；每个 strip 的 TE 边配对是否在 y 两侧用了同一顺序。
2. 修好后加回归断言 `max|AIC_mu - AIC_mu[m,m]| / max|AIC_mu| < 1e-12`（当前 1.66e-05）。
3. 生产求解器继续禁止对称化；半模只能作加速手段，验收基准仍用全机。

## 6. 产物

- `runs/backend_probes/aic_symmetry.json`、`aic_symmetry.log`
- `runs/backend_probes/aic_symmetry_matrices.npz`（body/wake/source/full/rhs/centers/normals/mirror）
- `runs/backend_probes/aic_symmetry_attribution.json`（逐块对称化的半模误差表）
- 辅助脚本：`runs/pr8_probe2.py`（附加导出+行级统计）、`runs/pr8_attribute.py`（逐块归因）
