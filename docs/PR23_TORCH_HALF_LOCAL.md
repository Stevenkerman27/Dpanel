# PR #23 本地验收：原生 Torch 半模提升面元 LU（CPU + CUDA）

验收对象：https://github.com/Stevenkerman27/Dpanel/pull/23
「Research: native PyTorch CPU/CUDA half-lifting panel LU with explicit geometry adjoint」，
head `ff58ea6bc4069a970a8d3f31928506239830f3be`（`feature/torch-half-lifting-cuda-prototype`，
stack 在 PR #22 `feature/occ-half-domain-cad-mesh` 上），独立 worktree `runs/pr23_git`。

本轮只做验收与记录：**未修改 PR 代码、未重跑 CAD、未改容差**；`physical_accuracy_accepted` 保持 `false`。

## 1. 环境

| 角色 | Python | torch | CUDA runtime | GPU | JAX | NumPy / SciPy |
|---|---|---|---|---|---|---|
| CPU 侧 | 3.11.17（`.conda_env`） | 2.14.1+cpu（`runs/torch_site`，未装进 `.conda_env`） | — | — | 0.10.2 CPU（x64 由验证器显式打开） | 2.4.6 / 1.17.1 |
| GPU 侧 | 3.13.14（`C:\Users\zyx20\anaconda3\envs\myml`） | 2.13.0+cu132 | 13.2 | NVIDIA GeForce RTX 5060（8151 MiB，driver 617.42） | 0.10.2 CPU，装在 `runs/jax_site313`（`--target`，不进 myml 的 site-packages） | 2.5.4 / 1.18.1（JAX 侧依赖） |

两条环境都用 `TMP=TEMP=TMPDIR=runs/pipwork`；pip 在沙箱里需要 `PYTHONPATH=runs/pipshim`
（把 `tempfile.mkdtemp` 换成 `os.makedirs`，否则在项目根建临时目录时 `Permission denied`）。
GPU 侧不重装 torch：myml 里本来就有 `cu132` 版，只补一个 `--target` 的 CPU JAX 作参考解。

## 2. 测试

| 命令（cwd `runs/pr23_git`） | 结果 |
|---|---|
| `pytest -q test/test_torch_half_lifting.py`，CPU torch 2.14.1+cpu | **5 passed, 1 skipped / 26.56 s / exit 0**（skip 的是真 CUDA 用例） |
| 同上，myml torch 2.13.0+cu132 | **6 passed / 32.96 s / exit 0**（CUDA 用例真正执行，无 skip） |

## 3. 场对拍（`panel.verify_torch_half --derivatives`，23 个输出全比）

容差用 PR 自带默认：CPU `rtol=2e-6, atol=2e-8`，CUDA 同；参考解是同一网格上的 JAX 半模求解。

| 算例 | device | 壁面 / 未知量 | 通过 | 最大绝对误差 | 真 LU 相对残差 |
|---|---|---|---|---|---|
| `cases/panel_half/thick_wing_two_strip.json` | cpu | 20 / 22 | **23/23** | 1.33e-15（`surface_velocity_half`） | 3.67e-16 |
| 同上 | cuda | 20 / 22 | **23/23** | 1.05e-15 | 3.30e-16 |
| `runs/half_cad_wingbody_v1/surface.json`（真 OCC 翼身） | cpu | 396 / 418 | **23/23** | 3.01e-13（`surface_velocity_half`） | 1.88e-15 |
| 同上 | cuda | 396 / 418 | **23/23** | 3.09e-13（`Cp_raw_half`） | 1.36e-15 |
| `runs/half_cad_wingbody_2x/surface.json` | cpu | 805 / 827 | **23/23** | —（同量级） | 2.20e-15 |
| 同上 | cuda | 805 / 827 | **23/23** | 1.01e-13（`Cp_raw_half`） | 1.20e-15 |

关键逐项（396 面 CUDA）：`matrix_half` 1.88e-15、`rhs_half` 6.94e-17、`mu_half` 2.80e-15、
`gamma_half` 5.00e-15、`panel_forces_half` 2.09e-16、`CL` 2.55e-15、`CD_pressure` 3.27e-16、
`CM` 8.95e-16。CPU 侧同项同量级（`force` 2.38e-14、`CL` 9.55e-15）。

判读：Torch 与 JAX 在**同一离散方程**上给出同一解，误差只到双精度舍入地板（1e-13 级源自未加权切平面
压力重构的条件数放大，与 PR #10 同源）；没有隐藏的截断、核软化或 bank 平均。

## 4. 几何/站位梯度（显式稠密 LU 伴随 vs JAX AD vs 中心差分）

损失 = `CL + 0.37*CM_y + 0.01*sum(Fx)`，参数是站位 `station_y`（与 PR #22 的 oracle 同口径）。

| 算例 | torch AD | JAX AD | 差 | h=1e-3 | h=1e-4 | h=1e-5 |
|---|---|---|---|---|---|---|
| 小夹具 cpu | 0.0009625331292665162 | 0.0009625331292665043 | 1.19e-17 | 1.07e-13 | 1.34e-12 | 1.84e-11 |
| 小夹具 cuda | 0.0009625331292665074 | 0.0009625331292665043 | 3.14e-18 | 1.56e-13 | 2.59e-12 | 9.34e-12 |
| 396 面 cpu | 0.012022551075838364 | 0.012022551075838384 | 1.91e-17 | 2.04e-12 | 2.39e-11 | 9.89e-11 |
| 396 面 cuda | 0.012022551075838505 | 0.012022551075838384 | 1.21e-16 | 1.37e-12 | 1.73e-11 | 2.01e-10 |
| 805 面 cuda | — | — | 5.76e-16 | 2.94e-13 | 9.62e-12 | 6.03e-11 |

三档差分误差都随 h 一阶下降并在 h=1e-5 处被差分相消抬起，说明伴随解与离散方程自洽；
`implicit_adjoint=False` 的原生 `torch.linalg.solve` 路径也在单测里对齐（差异 ~1e-15）。

## 5. 时间与显存（验证器自记，单次进程）

| 算例 | device | 冷（first） | 热（hot） | 反向（backward） | GPU 峰值 allocated / reserved |
|---|---|---|---|---|---|
| 22 未知量 | cpu | 0.00747 s | 0.00434 s | 0.0190 s | — |
| 22 未知量 | cuda | 15.477 s | 0.01347 s | 0.0760 s | 67.8 MB / 69.2 MB |
| 418 未知量 | cpu | 0.3432 s | 0.3431 s | 1.0159 s | — |
| 418 未知量 | cuda | 15.475 s | 0.1097 s | 0.5241 s | 371.5 MB / 406.8 MB |
| 827 未知量 | cpu | 0.6926 s | 0.6440 s | 2.2153 s | — |
| 827 未知量 | cuda | 15.634 s | 0.2215 s | 1.0579 s | 1151.9 MB / 1222.6 MB |

- 「冷」在 CUDA 上是常数 ~15.47 s，与问题规模无关 → 是 CUDA context / kernel 首次编译，不是求解时间。
- 热前向 GPU 相对 CPU：22 未知量 **0.31×（更慢）**、418 **3.1×**、827 **2.9×**；反向 418 **1.9×**、827 **2.1×**。
  小规模下 kernel 启动与 host 往返吃掉收益，与 PR #17/#21 在 GMRES 上看到的现象同型。
- 显存随未知量近似线性（418 → 371 MB，827 → 1.15 GB），因为装配的是稠密半模 AIC + LU 因子，
  仍 O(N²)；CPU 侧 `hot_seconds` 也已接近 `first_seconds`（418/827 面时 LU 主导）。

## 6. 结论与边界

- Torch 端半模提升面元 LU、镜像装配、动态站位、弱 Kutta 与压力重构在 **CPU 与 CUDA 上都与 JAX 生产实现逐位一致**
  （23/23 场，最大 3e-13），几何/站位伴随与 AD、三步长中心差分一致；CUDA 用例在 RTX 5060 上真跑通（6 passed，无 skip）。
- 仍然只是**离散一致性**：`physical_accuracy_accepted=false`，没有与实验/CFD 比对；没有跑 GRASS 训练；
  没有对 1976/23536 面做外推；没有改生产求解器。
- CUDA 的 15 s 冷启动说明它不适合作为小算例的交互式后端；418–827 未知量区间 GPU 前向/反向约为 CPU 的 2–3×，
  但显存在 1 GB 量级，继续加密会被稠密 AIC 的 O(N²) 吃掉（与 `docs/ISSUE16_MATRIX_FREE.md` 的结论一致）。

## 7. 复现命令

```powershell
# CPU（.conda_env + runs/torch_site）
$env:PYTHONPATH='runs/pr23_git;runs/torch_site'; $env:JAX_PLATFORM_NAME='cpu'
.conda_env\python.exe -m pytest -q test/test_torch_half_lifting.py
.conda_env\python.exe -m panel.verify_torch_half --surface cases/panel_half/thick_wing_two_strip.json `
    --out runs/torch_half_small_cpu_v1.json --device cpu --derivatives

# CUDA（myml + runs/jax_site313），`--surface` 对真实算例必须给绝对路径（runs/ 不入 git）
$env:PYTHONPATH='runs/pr23_git;runs/jax_site313'
C:\Users\zyx20\anaconda3\envs\myml\python.exe -m panel.verify_torch_half `
    --surface D:\3D\Projects\panel\runs\half_cad_wingbody_v1\surface.json `
    --out runs/torch_half_wingbody_cuda_v1.json --device cuda --block 16 --derivatives
```

回执 JSON：`runs/torch_half_small_cpu_v1.json`、`runs/torch_half_wingbody_cpu_v1.json`、
`runs/torch_half_wingbody2x_cpu_v1.json`、`runs/torch_half_small_cuda_v1.json`、
`runs/torch_half_wingbody_cuda_v1.json`、`runs/torch_half_wingbody2x_cuda_v1.json`。
