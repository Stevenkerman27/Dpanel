# 核函数分块：把稠密面元组装的内存从 O(N²) 降到 O(N·block)

日期 2026-10-25。改动文件：`panel/kernel.py`（`triangle_potential`）。API、签名、返回结构不变。

## 问题

`triangle_potential` 第一行 `r = triangles[None] - targets[:, None, None, :]` 直接 materialize **N×N×3×3**：
1976 面元时单个数组 1976²×9×8 B = **281 MB**，加上 `radius`、`determinant`、`denominator`、`omega`、`height`
以及每条边的 `cross`（94 MB）等约十个同阶数组，反向再存一份 cotangent → 实测峰值 4.17 GB（VJP 路径）/ 9.15 GB（生产 `--derivatives`）。

## 改法

目标点按块处理，块内用 `jax.lax.map` + `jax.checkpoint`：

```
body = jax.checkpoint(one_block) if _CHECKPOINT else one_block
stacked = jax.lax.map(body, jnp.arange(0, padded_count, block))
```

- 每个目标点的算术完全不变，只是循环结构变化；对角掩码由 `jnp.eye` 改成 `rows[:, None] == triangle_index[None, :]`（方形网格下等价）。
- 末块用 `jnp.repeat(targets[-1:], padding)` 补齐后截断，块大小不整除时也不丢项。
- 环境变量：`PANEL_KERNEL_BLOCK`（默认 **64**，0 关闭）、`PANEL_KERNEL_CHECKPOINT`（默认 **1**）。

**关键教训**：只加 `jax.lax.map` 不加 `jax.checkpoint` 会**更差** —— `lax.map` 编译成 `scan`，反向默认保存每轮残差，
实测 1976 面 block=256 无 checkpoint 是 7560 MB / 12.9 s（比不分块还高）；加 checkpoint 后 855 MB / 11.9 s。

## 数值等价性

| 检查 | 结果 |
|---|---|
| kernel 层：doublet（1976 面，block 0 vs 1024/512/256/128） | **逐位相同，max abs 0.0** |
| kernel 层：single | max abs **2.212e-17** |
| 生产 `fields.npz`：block=0 vs 已保存参考 | **逐位相同（0.0）** |
| 生产 `fields.npz`：block=64/256 vs 参考 | `Cp_raw` **4.79e-14**、`surface_velocity` 4.45e-14、`ql` 4.41e-14 |
| `CL` | 15 位全同 **0.319644907286278** |
| `panel/verify.py`（主矩阵 + 拒绝用例） | **34/34 passed**，`status: passed`（`runs/panel_verify_block64`） |

差异来源是分块求和顺序带来的舍入（single 层 2e-17 量级），不是近似或截断：没有加正则化、没有降精度、没有改核函数。

## 实测（真实 1976 面元算例 `runs/panel_error_budget_v2/aft_1.0001_size_0.15/solve`）

| 路径 | block | checkpoint | 峰值 RSS | 墙钟 |
|---|---|---|---|---|
| 前向 `panel/solve.py` | 0（原始） | — | **2768.6 MB** | 6.13 s |
| 前向 | 256 | 0 | 770.1 MB | 6.34 s |
| 前向 | 64 | 0 | 629.8 MB | 6.27 s |
| 前向 | 64 | 1（默认） | **631.2 MB** | 6.53 s |
| `--derivatives`（jacfwd） | 0（原始） | — | **9146.6 MB** | 24.14 s |
| `--derivatives` | 64 | 0 | 1202.9 MB | 25.62 s |
| `--derivatives` | 64 | 1（默认） | **1200.2 MB** | 26.46 s |
| `--derivatives` | 32 | 0 | 1091.7 MB | 25.60 s |
| VJP（PR #13 反向路径） | 0 | — | **4174.7 MB** | 9.95 s |
| VJP | 32 | 1 | **584.4 MB** | 11.79 s |
| VJP | 64 | 1 | 612.0 MB | 11.80 s |
| VJP | 128 | 1 | 700.2 MB | 12.19 s |
| VJP | 256 | 1 | 855.4 MB | 11.87 s |

即：**前向 4.4×、导数 7.6×、VJP 7.1× 内存下降**，时间代价 3–18%（前向基本免费，反向模式要重算一次组装）。

## 扩展性（`runs/kernel_scale.py`，真实 kernel，前向 + `value_and_grad`）

| n | block=0 | block=64 | 倍数 |
|---|---|---|---|
| 1000 | 891.4 MB | 354.8 MB | 2.5× |
| 2000 | 3136.1 MB | 504.8 MB | 6.2× |
| 4000 | **11847.6 MB** | **1049.3 MB** | 11.3× |

不分块 1000→2000 ×3.52、2000→4000 ×3.78（≈O(N²)）；分块 2000→4000 ×2.08（≈O(N)）。
时间代价在 n=4000 时约 16%（4.80 s → 5.56 s）。

## 现在的墙在哪

kernel 内存修好后，1976 面元的剩余占用主要是**稠密 AIC 矩阵本身**（1976²×8 B = 31 MB，在这一档还小）；
按 O(N²) 外推到 23536 面元：矩阵 **4.43 GB**，加 LU 因子再一倍 → **8.9 GB**，仍然放不进 8 GB 显卡；
`--derivatives`（jacfwd 三个参数）还要再乘参数份数。所以：

1. 内存这条线下一刀必须切**稠密矩阵/求解**（GMRES 或分块低秩/多极），不是 kernel；
2. 若只要"能跑"，23536 面元在 CPU 上按当前实现约需 9 GB 以上，不建议；
3. 精度方面本次改动是中性的（Cp 差 4.8e-14），没有改善压力场重构误差（−1.9%…−7.8% 仍在）。

## 未覆盖

- 只测 CPU float64；GPU 上 XLA 的 buffer 复用策略不同，需要重测。
- 未做多设备/多进程分块（`lax.map` 是单设备串行调度）。
- 未改 `panel/audit.py`（numba 独立核，内存本来 O(N)）。
- 未做稀疏/迭代求解，稠密 `jnp.linalg.solve` 与 `lu_factor` 原样保留。

## 产物

`panel/kernel.py`（改动）；`runs/kernel_ab.py`（逐位 A/B）、`runs/kernel_scale.py`（扩展性）、
`runs/solve_peak.py`（生产峰值 RSS wrapper）、`runs/vjp_peak.py`（VJP wrapper）、`runs/compare_prod.py`（fields 对比）；
验证输出 `runs/panel_verify_block64/`；配置扫描 `runs/cfg_64_ck{0,1}_{fwd,deriv}/`。
（2026-10-26 清理：这两类输出已移到 `archive/runs-2026-10-26/` 与 `archive/runs-2026-10-26/_bench_kernel/`，可用 `panel/verify.py` 与 `runs/kernel_scale.py` 重新生成。）
