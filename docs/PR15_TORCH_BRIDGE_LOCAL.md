# PR #15 本地验收：CPU Torch↔JAX 面元载荷桥

日期 2026-10-25，head `49f5b36d73a29b52ca1d41ffb24d3bbfdb0904af`（`feature/torch-jax-panel-bridge`，base `feature/panel-vertex-load-vjp` = PR #13 `9d68446`）。
独立工作树 `runs/pr15_work`；torch 来自隔离目录 `runs/torch_site`（`pip --target`），未改 `.conda_env`、未改依赖锁文件；未重跑 CAD、未覆盖任何旧 run。

## 环境

| 项 | 值 |
|---|---|
| Python | 3.11.17 |
| Torch | 2.14.1+cpu（CUDA 不可用，`torch.cuda.is_available()` = False） |
| JAX | 0.10.2，CPU，x64 打开（`float64`） |
| NumPy | 2.4.6 |
| 组合方式 | `PYTHONPATH=runs/pr15_work;runs/torch_site`，同一进程同时 import torch/jax |

## 命令与退出码

```
python -m pytest -q test/test_torch_panel_bridge.py                                  # 8 passed, 13.57 s, exit 0（云端 24.96 s）
python -m panel.verify_torch_bridge --run <saved solve> --out runs/<new> --derivatives # 4 例全 exit 0
```

八个用例（全 PASSED）：
`test_forward_and_all_seven_vjps_match_native_jax`、`test_torch_geometry_parameters_pass_gradcheck_and_three_step_fd`、
`test_gradient_reaches_torch_neural_network_parameters`、`test_partial_outputs_noncontiguous_tensors_and_inference`、
`test_two_inflight_calls_and_repeated_backward_are_independent`、`test_input_mutation_is_not_silently_used_for_backward`、
`test_double_backward_is_explicitly_unsupported`、`test_invalid_precision_shape_flow_and_geometry_are_rejected`。

## 前向：六输出与原生 JAX 一致（必须读作"同一路径的搬运"）

四个真实已保存构型上 `Cp_raw / 面力 / 面矩 / CL / CD_pressure / CM` 的 max abs 差全部 **0.0**。
这是**构造性相同**（桥内部调用同一个 `solve_fields`，只做 host copy），只能证明没有引入单位/转置/符号错误，
不能当作独立物理验证。真正的独立检查是 pytest 里"七输入 VJP vs 原生 JAX VJP"与下面的人工差分。

## 端到端参数链：七维 θ 的 Torch 反向

θ = (弦向缩放, 展向缩放, 后缘厚度鼓包, 迎角, 速度, 密度, 力矩原点 x 偏移)，种子为归一化随机权重（rng 419）。

| case | 壁面 | 冷前向 | 热前向 | 反向首个 | 反向热重算 | 墙钟 | 峰值 RSS |
|---|---|---|---|---|---|---|---|
| single_wing | 484 | 0.568 s | 0.0764 s | 1.765 s | 0.1014 s | 6.27 s | 860.1 MB |
| forward_body | 696 | 0.582 s | 0.1100 s | 1.473 s | 0.1486 s | 6.75 s | 1090.7 MB |
| two_bodies_coarse | 616 | 0.586 s | 0.1106 s | 1.761 s | 0.1397 s | 6.96 s | 1016.8 MB |
| aft_1.0001_h0.15 | 1976 | 1.054 s | 0.5716 s | 2.823 s | 1.062 s | 18.48 s | 4530.5 MB |

七维 Torch 梯度（`torch.autograd.grad`，与热重算的梯度 assert_close rtol/atol 1e-11 通过）：

| case | ∂/∂θ（7 项） |
|---|---|
| single_wing | −9.751e-4, 8.145e-5, 8.520e-4, 7.651e-4, −5.159e-5, −3.869e-5, −6.848e-3 |
| forward_body | 1.985e-4, −4.656e-3, 2.932e-3, −2.781e-3, 7.632e-5, 5.724e-5, −3.015e-3 |
| two_bodies_coarse | 4.928e-3, −1.269e-3, 2.007e-3, −9.332e-3, 7.216e-4, 5.412e-4, 1.337e-3 |
| aft_1.0001_h0.15 | −1.884e-2, 5.173e-3, 6.171e-3, 2.467e-2, −5.629e-5, −4.222e-5, 5.393e-3 |

两个方向 × 三步长中心差分（全部 passed，判据 `|ad−fd| ≤ 1e-7 + 2e-5|ad|`）：

| case | 方向 1 相对差 h=1e-3/1e-4/1e-5 | 方向 2 相对差 |
|---|---|---|
| single_wing | 2.37e-10 / 2.35e-09 / 3.28e-07 | 5.81e-10 / 1.79e-08 / 7.52e-09 |
| forward_body | 4.36e-09 / 1.57e-08 / 2.49e-07 | 5.87e-09 / 4.52e-08 / 5.26e-07 |
| two_bodies_coarse | 7.83e-10 / 8.55e-09 / 3.84e-07 | 4.38e-09 / 4.14e-08 / 3.61e-08 |
| aft_1.0001_h0.15 | 1.41e-07 / 1.19e-06 / 2.55e-05 | 6.29e-09 / 9.06e-09 / 8.89e-07 |

1976 面那档的 h=1e-5 相对差升到 2.5e-5，是差分本身在 1976 面/小目标值下的相消噪声（绝对值 1.05e-8），不是 AD 错误；
同一算例 h=1e-3 仍有 1.4e-7。

## 观察

1. 反向**重算**策略（Torch ctx 只存输入）：热反向 ≈ 热前向 + VJP 回代，484–616 面约 0.10–0.15 s，1976 面 1.06 s；
   没有出现 ctx 长持稠密 pullback 导致的峰值膨胀（1976 面 4.53 GB，与 PR #13 纯 JAX 的 4.18 GB 同量级）。
2. 冷启动成本（0.57 s / 1.05 s）主要是 JAX 编译，训练循环里可摊薄；每个新网格/拓扑仍要重编译。
3. 反向成本与前向同阶，说明"重算"没有付费太多；若改为在 ctx 保留 pullback，内存会明显上升——本轮未测，不建议默认打开。

## 明确不支持 / 未覆盖

- 仅 CPU float64；无 CUDA、DLPack、零拷贝声明。
- 无 `vmap`/批量、无 `torch.compile`、无二阶反向（`once_differentiable` 显式拒绝）。
- 仅固定拓扑；拓扑/接触变化必须重建夹具，非法几何在 import 前被拒。
- 未接 GRASS 的 CST/机身真实参数化；本轮 θ 是解析形变，不是生成器参数。
- `physical_accuracy_accepted=false` 保持；这是训练**接口**验收，不是气动精度验收，也未启动正式训练。

## 产物

`runs/torch_bridge_{single_wing,forward_body,two_bodies_coarse,aft_fine_1976}/report.json`；
工具 `runs/bridge_peak.py`（峰值 RSS wrapper）、`runs/run_bridge_verify.ps1`、`runs/run_bridge_1976.ps1`。