# PR #17 本地验收：矩阵无关 GMRES / 近场预条件 / 严格镜像半模

对 PR #17 分支 `feature/matrix-free-gmres-symmetry` head
`14344e3f3d80c1eaa2cc037db88248a4b9f2af81`（base `49f5b36d`，即 PR #15 的 Torch 桥）做的本地实测。
所有数字来自本机独立进程、真实保存场（`runs/panel_robust_matrix_v1/{single_wing,two_bodies_coarse,forward_body}`），
不使用合成 fixture，不重跑 23536 面元，不改生产代码。`physical_accuracy_accepted` 保持 `false`。

## 1. 环境

| 项 | 值 |
| --- | --- |
| 平台 | Windows，32 逻辑核，Python 3.11.17（`.conda_env`） |
| JAX | **0.10.2**（CPU，float64 开启；`jax_platform_name=cpu`） |
| SciPy / NumPy | 1.17.1 / 2.4.6 |
| torch | 2.14.1+cpu（仅 `runs/torch_site`，本 PR 不依赖） |
| 代码 | worktree `runs/pr17_work` @ `14344e3f`，`PYTHONPATH=runs/pr17_work;runs/torch_site` |
| 沙箱 | `TMP/TEMP/TMPDIR=runs/pipwork`、`NUMBA_CACHE_DIR=runs/numba_cache` |

## 2. 单元测试（JAX 0.10.2）

| 命令 | 结果 | 耗时 | 退出码 |
| --- | --- | --- | --- |
| `python -m pytest -q test/test_iterative_jax010_compat.py` | 3 passed | 15.65 s | 0 |
| `python -m pytest -q test/test_iterative_panel.py test/test_differentiable_panel_fields.py` | 16 passed | 37.69 s | 0 |

首轮在 head `1c43a62` 上本地 collection 直接失败：`panel/iterative.py:17`
`from jax.custom_transpose import custom_transpose` → `ModuleNotFoundError`（JAX 0.10.2 无此模块）。
云端在 `7a66f581` 加了能力检测后备：无 `custom_transpose` 时走 `jax.pure_callback` + SciPy GMRES +
`jax.custom_vjp` 一阶隐式伴随，`4bc17337` 又加 `test_iterative_jax010_compat.py` 强制覆盖该路径。
下面的所有耗时都来自这条**后备路径**，不是 JAX-native 迭代求解。

## 3. 四模式基准（block=32 / near=32 / restart=40 / cycles=20 / tol=1e-10）

每个单元格都是独立进程 + 独立输出目录；峰值 RSS 由 `runs/pr17_peak.py` 用
`K32GetProcessMemoryInfo(...).PeakWorkingSetSize` 每 20 ms 采样（`verify_iterative.py` 自带的
`process_peak_rss_bytes` 在本机为 `null`）。

| 算例 | 模式 | 未知量 | 冷前向 | 热前向 | 冷 VJP | 热 VJP | 真实残差 | 峰值 RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_wing | LU | 523 | 0.529 s | 0.076 s | 2.003 s | 0.174 s | 1.05e-15 | 511.9 MB |
| single_wing | GMRES 无 PC | 523 | 9.018 s | 11.595 s | — | — | 1.00e-10 | 527.3 MB |
| single_wing | GMRES + 近场块 PC | 523 | 5.388 s | 1.134 s | 5.791 s | 3.580 s | 2.43e-11 | 902.4 MB |
| single_wing | 半模 | — | — | — | — | — | — | 177.1 MB（拒绝） |
| two_bodies_coarse | LU | 651 | 0.557 s | 0.121 s | 2.123 s | 0.290 s | 1.11e-15 | 533.0 MB |
| two_bodies_coarse | GMRES 无 PC | 651 | 10.705 s | 12.678 s | — | — | 9.76e-11 | 524.9 MB |
| two_bodies_coarse | GMRES + 近场块 PC | 651 | 5.806 s | 1.715 s | 7.015 s | 4.369 s | 3.73e-11 | 898.8 MB |
| two_bodies_coarse | 半模 | — | — | — | — | — | — | 177.7 MB（拒绝） |
| forward_body | LU | 735 | 0.554 s | 0.131 s | 2.140 s | 0.287 s | 9.14e-16 | 531.5 MB |
| forward_body | GMRES 无 PC | 735 | 15.229 s | 18.902 s | — | — | 9.11e-11 | 533.3 MB |
| forward_body | GMRES + 近场块 PC | 735 | 5.973 s | 1.711 s | 7.919 s | 5.462 s | 2.28e-11 | 906.3 MB |
| forward_body | 半模 | — | — | — | — | — | — | 178.5 MB（拒绝） |

反直觉但可复核的两点：（a）**预条件后的 GMRES 峰值 RSS 反而更高**（≈900 MB vs LU 511 MB），
因为后备路径要同时持有 SciPy Krylov 基、近场块 LU 因子和 JAX 运行时缓冲；（b）**LU 冷/热前向快一个数量级**
（0.53 s vs 5.4 s，冷；0.08 s vs 1.1 s，热）。在这个规模上迭代法没有任何速度或内存优势，
它的价值只在"不形成完整 AIC"这一条，需要更大的 N 才能兑现，而本机不能拿 23536 面元验证。

## 4. 数值一致性（对存盘场 `fields.npz` / `result.json`）

断言阈值来自 `panel/verify_iterative.py:83`（`rtol=3e-6, atol=2e-8`），全部通过；下表是相对 L2。

| 算例 | 模式 | Cp_raw | panel_forces | CL | mu | gamma | CM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single_wing | LU | 1.4e-14 | 1.8e-15 | 1.1e-16 | 1.2e-15 | 1.5e-15 | 9.0e-15 |
| single_wing | GMRES + PC | 2.8e-10 | 6.7e-11 | 5.9e-12 | 4.4e-11 | 5.1e-11 | 7.3e-10 |
| single_wing | GMRES 无 PC | 5.3e-09 | 1.2e-09 | 2.1e-09 | 1.8e-09 | 2.0e-09 | 5.9e-09 |
| two_bodies_coarse | LU | 9.0e-15 | 1.3e-15 | 2.1e-16 | 9.7e-16 | 1.3e-15 | 1.4e-14 |
| two_bodies_coarse | GMRES + PC | 9.3e-10 | 1.3e-10 | 5.1e-11 | 9.3e-11 | 1.1e-10 | 1.4e-09 |
| two_bodies_coarse | GMRES 无 PC | 3.8e-09 | 1.1e-09 | 3.4e-09 | 2.0e-09 | 2.3e-09 | 5.4e-09 |
| forward_body | LU | 1.5e-14 | 2.0e-15 | 1.4e-15 | 1.6e-15 | 2.2e-15 | 1.9e-14 |
| forward_body | GMRES + PC | 6.7e-10 | 1.1e-10 | 1.7e-11 | 6.8e-11 | 8.9e-11 | 1.1e-09 |
| forward_body | GMRES 无 PC | 7.1e-09 | 2.6e-09 | 6.1e-09 | 3.5e-09 | 4.0e-09 | 1.2e-08 |

载荷误差比状态误差大约一个数量级，和 issue #16 里查明的切平面模板病态（`single_wing` 模板行列式最小
1.9e-06 → 放大 5.25e+05）一致；这是压力重构的模型误差，不是线性代数失败。

## 5. 设计方向 VJP 与三档中心差分

方向：`dx = chord*(0.02*xn0 + 0.005*xn1^2, 0.01*xn1^3/(1+max(yn^2)), 0.02*xn2)`，`dy = chord*0.01*yn^3/(1+max(yn^2))`，
目标 `sum(w*panel_forces) + 0.2*sum(v*panel_moments)`（`rng=20261025`），AD 与三档 `h` 的绝对差：

| 算例 | 模式 | AD | h=1e-3 | h=1e-4 | h=1e-5 |
| --- | --- | --- | --- | --- | --- |
| single_wing | LU | 0.00345725037 | 1.83e-13 | 1.94e-12 | 4.85e-11 |
| single_wing | GMRES + PC | 0.00345725037 | 1.40e-12 | 6.17e-12 | 2.36e-11 |
| two_bodies_coarse | LU | -0.00667390077 | 1.65e-12 | 2.61e-12 | 2.50e-13 |
| two_bodies_coarse | GMRES + PC | -0.00667390077 | 6.53e-12 | 1.50e-13 | 7.88e-11 |
| forward_body | LU | -0.00386378052 | 4.04e-13 | 7.23e-13 | 4.06e-11 |
| forward_body | GMRES + PC | -0.00386378052 | 1.25e-12 | 5.36e-12 | 6.33e-12 |

迭代路径的伴随（转置求解）与 LU 路径给出同一方向导数（同 9 位以上），说明隐式伴随转置实现与正向一致；
无 PC 路径未做 `--derivatives`（每次 FD 评估都要重跑一整次未预条件 GMRES，代价 ~6x30 s 以上，没有额外信息量）。

## 6. 半模：三例全部 blocked（严格拒绝）

`--half` 在三个真实网格上都在建模阶段被门控拒绝，退出码 1，未做任何平均/修补/重划网格：

`ValueError: Need bijective mirrored vertices, including wake nodes`（`panel/iterative.py:101`，`make_plan`）

| 算例 | 退出码 | 耗时 | 峰值 RSS |
| --- | --- | --- | --- |
| single_wing | 1 | 0.9 s | 177.1 MB |
| two_bodies_coarse | 1 | 0.87 s | 177.7 MB |
| forward_body | 1 | 0.92 s | 178.5 MB |

这与 issue #16 里我自己测到的结论一致：这五个 Gmsh 网格站位是对称的，但 OCC 三角化左右不镜像
（`single_wing` 484 面里只有 26 个面元有精确镜像伙伴，最近镜像距离中位 1.33e-2 弦长），
所以 `R A E` 前提不成立，只能标 blocked。要真正打开半模，需要半边建模 + 镜像生成或对称感知的重网格，
不能靠放宽门控。

## 7. 判读

1. 云端这版实现在 JAX 0.10.2 上可用（走 SciPy 后备），数值上确实收敛到真残差 1e-11 量级，
   状态/载荷/梯度都与 LU 一致到 1e-10…1e-9，没放宽任何阈值。
2. 在本机能跑的三个规模上，迭代法**没有**带来内存或时间收益（峰值 RSS 900 MB vs 511 MB，
   前向慢 10 倍）；唯一成立的结论是"不形成完整 AIC 也能解出同样精度的场与伴随"。
   要把它变成 23536 面元的现实选项，必须在 JAX 0.9+ 原生 `custom_transpose` 路径上重测，
   并且证明迭代数随 N 基本不变。
3. 半模在这个仓库的网格上不可用（严格拒绝），这是网格生成的问题，不是降阶数学的问题；
   VortexAD 的 1616 面 STL 上 `R A E` 与 `A_RR + A_RL` 逐位相同（0.0 差），说明公式实现没问题。
4. `physical_accuracy_accepted` 保持 `false`；本次只验证"线性代数等价 + 伴随正确 + 拒绝策略"，
   不涉及物理精度。

## 8. 复现

```powershell
$env:PYTHONPATH="D:\3D\Projects\panel\runs\pr17_work;D:\3D\Projects\panel\runs\torch_site"
$env:JAX_PLATFORM_NAME="cpu"; $env:TMP=$env:TEMP=$env:TMPDIR="D:\3D\Projects\panel\runs\pipwork"
# 单模式，一个进程一个目录；--run 指已保存 solve 目录
python runs\pr17_peak.py <outDir> --run runs\panel_robust_matrix_v1\single_wing --out <outDir> `
  --mode gmres --pc block --block 32 --near 32 --restart 40 --cycles 20 --tol 1e-10 --derivatives
# 全矩阵（三算例 x {LU, GMRES+PC, GMRES, 半模}）
Invoke-Expression (Get-Content runs\pr17_bench.ps1.txt -Raw)
```

原始输出：`runs/pr17_bench/<case>_<mode>_<pc>/report.json`、`process_peak_rss_mb.json`、`runs/pr17_bench/driver.log`。
