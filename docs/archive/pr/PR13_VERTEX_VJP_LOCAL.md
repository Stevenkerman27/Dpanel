# PR #13 本地验收：顶点输入升力面元 + 隐式几何 VJP

日期 2026-10-25，head `9d6844611dde5b39f7d14dd313e2d45557e4b2d4`（`feature/panel-vertex-load-vjp`，base `local/panel-jax-20261025` @ `08215a6`）。
独立工作树 `runs/pr13_work`，未改生产 `panel/solve.py` / `aero.py` / kernel，未覆盖任何旧 run。补充：PR #12（`2469caff`，vertex 势值证明）也已跑，2 passed / 5.94 s / exit 0。

## 环境

| 项 | 值 |
|---|---|
| Python | 3.11.17（`.conda_env`） |
| JAX | 0.10.2，backend `CpuDevice(id=0)`，`x64_enabled=False` 默认，脚本内 `verify_differentiable.py:33` 显式 `jax.config.update("jax_enable_x64", True)`；`differentiable.py:68` 在未开 x64 时直接抛错 |
| NumPy / SciPy | 2.4.6 / 1.17.1（OpenBLAS） |
| 硬件 | 32 逻辑核，无 GPU 参与 |
| 沙箱注意 | `TMP/TEMP/TMPDIR=runs/pipwork`（否则 numba/pytest 临时文件路径报 WinError 5） |

## 命令与退出码

```
python -m pytest -q test/test_differentiable_panel_fields.py      # 6 passed, 14.34 s, exit 0
python -m panel.verify_differentiable --run <saved solve> --out <new> --derivatives   # 5 例全 exit 0
```

云端记录的 6 passed / 25.13 s（Python 3.13.5 / JAX 0.9.0.1）本地复现为 **6 passed / 14.34 s**（Python 3.11.17 / JAX 0.10.2）。
唯一警告是 pytest cache 目录在沙箱下 `WinError 5`，与断言无关。

## 数值回归：新接口 vs 已保存的生产场（逐位级一致）

| case | 壁面 | CL（新） | CL（原 result.json） | Cp_raw max abs | 面力 relL2 | 面矩 max abs | mu max abs | gamma max abs | CL max abs | 残差 |
|---|---|---|---|---|---|---|---|---|---|---|
| single_wing | 484 | 0.241534418 | 0.2415344184 | 2.56e-14 | 1.25e-15 | 6.94e-17 | 3.75e-16 | 6.66e-16 | 1.94e-16 | 1.04e-15 |
| forward_body | 696 | 0.269042927 | 0.2690429275 | 3.33e-14 | 1.43e-15 | 4.35e-17 | 8.60e-16 | 5.83e-16 | 2.78e-16 | 9.18e-16 |
| two_bodies_coarse | 616 | 0.263212956 | 0.2632129558 | 2.91e-14 | 9.42e-16 | 6.31e-17 | 5.27e-16 | 8.33e-16 | 5.55e-17 | 1.17e-15 |
| two_bodies_fine | 1646 | 0.312071654 | 0.3120716537 | 1.61e-14 | 1.79e-15 | 4.15e-17 | 5.00e-16 | 9.16e-16 | 6.11e-16 | 9.99e-16 |
| aft_1.0001_h0.15 | 1976 | 0.319644907 | 0.3196449073 | 2.09e-14 | 1.98e-15 | 2.43e-17 | 5.55e-16 | 9.99e-16 | 1.11e-16 | 9.25e-16 |

五例全部 `numerical_regression_passed`；`physical_accuracy_accepted=false` 未动。
说明：这是"新接口复现旧保存场"的一致性证明，不是独立物理精度验收（压力 CL 与环量 CL 仍差 −1.9%…−7.8%）。

## 几何 VJP：任意局部载荷种子

| case | 局部种子目标 | ‖dL/dX‖ | ‖dL/dy_station‖ | AD（h=1e-3 方向导数） | 相对误差 h=1e-3 / 1e-4 / 1e-5 |
|---|---|---|---|---|---|
| single_wing | −0.008827 | 5.2390 | 0.017069 | 0.004079669667084 | 1.04e-10 / 1.20e-09 / 1.29e-08 |
| forward_body | −0.155426 | 6.3409 | 0.033288 | −0.001839367989836 | 4.98e-10 / 3.62e-10 / 1.51e-08 |
| two_bodies_coarse | −0.309727 | 5.4035 | 0.039873 | −0.010967405712105 | 1.52e-10 / 2.43e-10 / 2.60e-09 |
| two_bodies_fine | 0.268859 | 5.5541 | 0.175659 | 0.012153464918525 | 8.08e-11 / 2.36e-10 / 1.51e-08 |
| aft_1.0001_h0.15 | −0.040031 | 5.5379 | 0.011380 | 0.001024284510685 | 1.09e-09 / 1.22e-09 / 1.13e-07 |

- VJP 是**反向模式**（`custom_linear_solve` + 显式转置回代，复用正向 LU），代价与种子个数无关；
- h=1e-5 时的退化（1e-7 级）符合中心差分的舍入地板，不是 AD 误差；
- 站位导数 `dL/dy_station` 也非零且量级合理（0.011–0.176），说明动态尾迹/TE 基函数确实进入了图。

## 时间与内存（`.conda_env`，单进程，父进程读 PeakWorkingSet64）

| case | 壁面 | 墙钟 | 峰值 RSS | 编译+首跑 | 热场 | VJP 编译+首跑 |
|---|---|---|---|---|---|---|
| single_wing | 484 | 3.83 s | 655.7 MB | 0.627 s | 0.0782 s | 1.637 s |
| forward_body | 696 | 3.92 s | 885.8 MB | 0.653 s | 0.1319 s | 1.361 s |
| two_bodies_coarse | 616 | 4.27 s | 814.9 MB | 0.668 s | 0.1237 s | 1.728 s |
| two_bodies_fine | 1646 | 8.04 s | 2942.7 MB | 0.991 s | 0.4691 s | 2.253 s |
| aft_1.0001_h0.15 | 1976 | 10.20 s | 4177.3 MB | 1.172 s | 0.6083 s | 2.770 s |

内存是硬约束：1976 壁面已到 4.18 GB（稠密 AIC + LU 因子 + VJP 中间量），按稠密 O(n²) 外推 23536 壁面不可行；
这与 PR #5 的结论一致（23536 的 A 阵单精度 4.13 GiB、LU 因子再翻倍）。GPU 8 GB 也放不下。

## 明确未覆盖（不得据此宣称可微 CAD）

1. CAD 布尔、重网格、接触/尾缘拓扑变化：全部未求导；`solve_fields` 需要固定拓扑，非法变形由 `validate_configuration` 拒绝。
2. 仅平面尾迹 + 固定 TE 基函数；自由/非平面尾迹未实现。
3. Torch↔JAX 桥（GRASS 参数化 → 节点 → dL/dX → dL/dθ）未做，本轮无跨框架零拷贝声明。
4. 无 GPU/DLPack、无批量吞吐测试；全部 CPU float64。
5. 物理精度仍为 false；本 PR 只证明"离散求解器对固定拓扑节点坐标的导数正确"，不证明气动载荷物理正确。

## 结论

- 生产场可在新接口下**逐位复现**（Cp 2–3e-14，面力/面矩 1e-15 级），说明抽取没有改变离散化；
- 几何 VJP 在 5 个真实构型（单翼、前置机身、粗/细双机身、近接触 1976 面）上与中心差分一致到 1e-10…1e-9；
- 与 PR #10 的结论不冲突：压力场重构误差依旧存在，VJP 只是把"导数通道"修好，不能把低阶压力场提升为已验证载荷。

## 产物

`runs/vertex_vjp_{single_wing,forward_body,two_bodies_coarse,two_bodies_fine,aft_fine_1976}/report.json`（含全部字段差、VJP 范数与三档差分）；
`runs/vjp_peak.py`（峰值 RSS wrapper）、`runs/run_vertex_verify.ps1`（批量）。