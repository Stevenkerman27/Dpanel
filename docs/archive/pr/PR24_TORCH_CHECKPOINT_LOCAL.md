# PR #24 本地验收：Torch 半模解析核的可选 rematerialization（CUDA A/B）

日期 2026-10-26。head `8df8492015ac0321ea42abf23987247dbc8b04b4`（`perf/torch-half-kernel-checkpoint`，stack 在 PR #23 上），worktree `runs/pr24_git`。

## 1. 环境

| 项 | 值 |
| --- | --- |
| Python | 3.13.14（`C:\\Users\\zyx20\\anaconda3\\envs\\myml\\python.exe`） |
| PyTorch | 2.13.0+cu132，`torch.cuda.is_available()=True` |
| CUDA runtime | 13.2，driver 617.42 |
| GPU | NVIDIA GeForce RTX 5060，8151 MiB |
| JAX | 0.10.2（`runs/jax_site313`，CPU 参考） |
| dtype | float64 全程（`float64: true`） |

## 2. 单测

```bat
$env:PYTHONPATH='...\\runs\\pr24_git;...\\runs\\jax_site313'
python -m pytest -q test/test_torch_half_core_cpu.py test/test_torch_half_lifting.py test/test_torch_half_checkpoint.py
```

**12 passed / 31.04 s / exit 0**，无 skip（CUDA 用例真实执行，未静默回退 CPU）。

## 3. A/B 实测（每档独立进程，`runs/peak_wrap.py` 20 ms 采样宿主峰值 RSS）

| 文件 | 模式 | 壁面/未知量 | first (s) | hot (s) | 真残差 | 场对比 | CUDA 峰值 alloc/reserved | 进程 RSS | backward (s) | 站位导数差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `torch_ckpt_off_418_v1.json` | OFF | 396 / 418 | 15.403 | 0.1112 | 1.357e−15 | 23/23 | 371 / 407 MB | 2817 MB | 0.52 | 1.21e−16 |
| `torch_ckpt_on_418_v1.json` | ON | 396 / 418 | 15.459 | 0.1121 | 1.357e−15 | 23/23 | **88 / 115 MB** | 2887 MB | 1.44 | 1.21e−16 |
| `torch_ckpt_off_827_v1.json` | OFF | 805 / 827 | 15.624 | 0.2636 | 1.201e−15 | 23/23 | 1152 / 1223 MB | 2916 MB | 1.23 | 5.76e−16 |
| `torch_ckpt_on_827_v1.json` | ON | 805 / 827 | 15.505 | 0.2249 | 1.201e−15 | 23/23 | **136 / 189 MB** | 2946 MB | 2.14 | 5.76e−16 |
| `torch_ckpt_off_1988_v1.json` | OFF | 1966 / 1988 | 15.969 | 0.5466 | 1.376e−15 | 23/23 | 5655 / 5778 MB | 3250 MB | 2.65 | 1.01e−15 |
| `torch_ckpt_on_1988_v1.json` | ON | 1966 / 1988 | 15.919 | 0.5533 | 1.376e−15 | 23/23 | **438 / 575 MB** | 3290 MB | 4.31 | 1.01e−15 |

三档 surface：`half_cad_wingbody_v1`（396 壁面）、`half_cad_wingbody_2x`（805）、`half_wb_size0p10`（1966）。命令：

```bat
python runs/peak_wrap.py panel.verify_torch_half <peakdir> \
  --surface <abs surface.json> --out runs/torch_ckpt_{off,on}_<N>_v1.json \
  --device cuda --block 16 --max-panels 2200 --derivatives [--checkpoint-kernel]
```

## 4. 判读

1. **数值上完全等价**：ON/OFF 的 23 个场逐项误差、真残差、几何与站位导数差值**逐位相同**（418 档 `Cp_raw_half` 3.09e−13、CL 2.55e−15；1988 档站位导数差 1.01e−15），且 23/23 均在 `rtol=2e-5, atol=1e-7` 内。rematerialization 只改变反向图的存储，不改变离散方程。
2. **显存收益随规模放大**：418 → 371→88 MB（−76%）；827 → 1152→136 MB（−88%）；**1988 → 5655→438 MB（−92%，13×）**。OFF 在 1988 档已经吃掉 5.7 GB，8 GB 卡上再叠加训练 batch 就会 OOM；ON 只占 0.44 GB。
3. **代价是反向时间**：418 档 0.52→1.44 s（2.8×）、827 档 1.23→2.14 s（1.7×）、1988 档 2.65→4.31 s（1.6×）；前向不变（变化 ≤1.2%）。
4. **进程 RSS 不变**（2817→3290 MB）：`torch_first_seconds` 恒定 ~15.5 s 且与规模无关 = CUDA context + 首次 kernel 编译；宿主侧常驻的是 Python/JAX/NumPy 的 2.8 GB，checkpoint 省的是 GPU 显存而不是宿主内存。

## 5. 建议

- 小规模（<800 壁面）保持默认 OFF：显存没有压力，反向更快。
- **≥800 壁面默认开 checkpoint**：805 档省 1.0 GB、1966 档省 5.2 GB，反向只慢 1.6–1.7×；本机 8 GB 卡上 1966 壁面（当前 5% 收敛推荐规模）只有 ON 模式留得住余量。
- 上游默认 OFF 可以接受；建议把开关接入 `solve_half_torch` 的自动策略（`half_wall_panels >= 800` 时默认 ON），并把实际模式写进 `report.json`。

## 6. 边界声明

- 只验证 396/805/1966 壁面三个真实半模，未跑 23536 面板全机。
- 未测 CUDA 显存碎片化与多进程并发。
- `physical_accuracy_accepted=false`：本报告只证明 A/B 在显存与时间上的取舍，不涉及物理精度。
