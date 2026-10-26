# PR #22 第三轮本地验收：自适应压力模板环扩展（真实翼身半模）

日期 2026-10-26 ｜ head `ce8eb5492d5078a09496aeace20c525bc0b29fdc` ｜ 独立 worktree `runs/pr22c_git`

## 0. 环境与命令

| 项 | 值 |
| --- | --- |
| Python | 3.11.17（`.conda_env`） |
| JAX | 0.10.2，CPU，`jax_enable_x64=True` |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| Gmsh | 4.15.2（本轮未重新生成网格） |
| 输入（绝对路径） | `runs/half_cad_wingbody_v1/surface.json`，SHA-256 `d04e439c…` |
| 沙箱 | `TMP/TEMP/TMPDIR=runs/pipwork`（沙箱里 `tempfile.mkdtemp` 目录不可写） |

## 1. 单元测试：21 passed / 1 sandbox error

```
pytest -q test/test_half_pressure_stencil.py test/test_half_surface_mesh.py \
        test/test_half_native_panel.py test/test_half_mirror_oracle.py
```

结果：**21 passed，1 error，EXIT=1**。唯一 error 是 `test_contact_root_station_is_one_shared_mirror_orbit` 的 setup：
`PermissionError: [WinError 5] … runs/pr22_bt4`，来自 pytest `tmp_path`（沙箱里 `mkdtemp` 建出的目录不可列/不可写），
会话结束时 `_pytest/pathlib.py:354 cleanup_dead_symlinks` 同样报 WinError 5，所以退出码是 1 而不是 0。
该用例的断言用 `runs/p22b_manual/run_contact_test.py` 单独执行：
`half_circulation_unknowns 1`、`full_oracle_circulation_unknowns 1`、`expanded_half_state_vs_full` max abs **9.71e−17** / rel L2 **5.70e−16** → 通过。
没有改容差、没有改 fixture、没有改求解代码；其余 21 个用例（含 6 个压力模板用例、2 个半模原生用例、镜像 oracle 的真实翼身用例）在真实代码路径上通过。

## 2. 原错误对照：`--rings 2 --max-stencil-rings 2`

```
runs/half_wb_rank2_reject_v1 : status=failed, EXIT 1, 峰值 153.3 MB
ValueError: Rank-deficient cut-aware pressure stencil face=238, initial_ratio=0.000e+00,
            last_ratio=0.000e+00, neighbors=4, last_depth=2, max_rings=2;
            do not cross true TE/contact cut or relax the rank gate
```

即：默认不扩展时仍以 face 238 拒绝，报错信息现在明确给出面号、初始/最终奇异比、可达邻居数与尝试深度 —— 比上一轮的 `stencil rank deficient` 可诊断。

## 3. 自适应扩展：`--rings 2 --max-stencil-rings 6`

`report.json.pressure_stencil_audit`：

| 项 | 值 |
| --- | --- |
| `initial_rank_deficient_faces` | **[238, 359]** |
| `expanded_faces` | **[238, 359]** |
| `initial_singular_ratios`（这两面） | 0.0 / 0.0 |
| `final_singular_ratios`（这两面） | **0.6945053626157747 / 0.7133032873506489** |
| `selected_ring_depth`（这两面） | **3 / 3** |
| 其余 394 面的深度 | 全部 2（`[(i,d) for ... if d != 2] == [(238,3),(359,3)]`） |
| 全网格最小最终奇异比 | 0.11411808496489085（396 面全 ≥1e−6） |
| `pressure_neighbor_width` | 9（与原先一致） |

与云端期望逐条一致：只有这两面被扩到第三环，不是六环；其余面元的两环邻接未被改动；门控阈值仍是 1e−6，没有放宽。

### LU（`runs/half_wb_rank6_lu_v1`）

| 项 | 值 |
| --- | --- |
| 未知量 | 418 = 396 壁面 μ + 22 γ（画幅独立） |
| 真实相对残差 | **9.887e−16** |
| 首次 / 热 | 0.6808 s / 0.08637 s |
| 峰值 RSS | **410.3 MB** |
| CL / CD_pressure / CM_y | 0.5312296424578451 / 0.006167551183545117 / −0.022266751896092483 |
| force_N / moment_Nm | [−0.09838129905838668, 0, 1.2978769424355368] / [0, −0.05455354214542659, 0] |
| `Cp_raw` 范围 | [−4.034247560990468, 0.6479920104052215] |
| status | `numerical_solve_completed_not_physical_validation` |

## 4. 同一网格的 GMRES / 块预条件（`runs/half_wb_rank6_gmres_v1`）

参数 `--block 32 --near 32 --restart 40 --cycles 20 --tol 1e-10`：

| 项 | 正向 | 转置 |
| --- | --- | --- |
| 迭代 / matvec | 28 / 30 | 29 / 31 |
| 真实相对残差（求解器内部） | 3.1137e−11 | 1.7932e−11 |
| 真实相对残差（独立重算 ‖Ax−b‖/‖b‖） | 3.1137e−11 | 1.7932e−11 |
| 耗时 | 首次 6.780 s / 热 1.652 s | 1.780 s（同进程测量） |
| 峰值 RSS | 824.6 MB | 759.7 MB（正+转置同一 harness） |

LU vs GMRES 载荷差：dCL **1.7e−11**、dCD 1.4e−12、dCM_y 7.3e−12。
伴随恒等式 `<Ax,y>` vs `<x,Aᵀy>` 差 **2.22e−16**（相对 2.02e−16），转置预条件路径的 `P⁻ᵀ` 也一并走过。
该规模（418 未知量）上流式 GMRES 仍比 LU 慢约 10×、峰值内存约 2×，**不据此声称任何加速或省内存**。

## 5. 精确镜像 oracle（`runs/half_wb_rank6_mirror_v1.json`）

`panel.verify_half_mirror --surface … --out … --derivatives --max-half-panels 1000` → **EXIT 0**，峰值 **671.8 MB**，
`status = exact_mirror_discretization_verified`。半模 396 壁面 / 22 γ / 418 未知量；oracle 全模 792 壁面 / 43 γ / 835 未知量。

| 量 | max abs | relative L2 |
| --- | --- | --- |
| `A_half` vs `A_full[R,:]E` | **0.0** | 0.0 |
| RHS | 0.0 | 0.0 |
| μ 半侧 vs 全模正侧 | 0.0 | 0.0 |
| γ 半侧 vs 全模正侧 | 0.0 | 0.0 |
| 展开状态 | 0.0 | 0.0 |
| 右半 raw Cp | 0.0 | 0.0 |
| 右半面力 / 面矩 | 0.0 / 0.0 | 0.0 / 0.0 |
| 整机压力力 / 矩 | 0.0 / 0.0 | 0.0 / 0.0 |
| CL / CM | 0.0 / 0.0 | 0.0 / 0.0 |

CL 半模 0.5312296424578451 vs oracle 0.5312296424578457。全部为零是构造性的：oracle 用完全相同的镜像网格与同一套自适应模板装配，
所以这里证明的是「半模装配 = 全模镜像装配」逐位成立，而不是两种实现独立一致（翼身之外的独立实现已在机翼算例与 PR #21 的合成夹具上验过）。

站位几何导数（`symmetric_geometry_station_derivatives`，方向 `station_y`，值 −0.0018975676272125392）：

| h | half AD | full AD | half FD 绝对误差 | full FD 绝对误差 |
| --- | --- | --- | --- | --- |
| 1e−3 | −0.0018975676272125392 | −0.0018975676272126188 | 3.69e−13 | 6.52e−14 |
| 1e−4 | 同上 | 同上 | 5.43e−12 | 3.07e−12 |
| 1e−5 | 同上 | 同上 | 5.05e−11 | 1.30e−11 |

半模 AD 与全模 AD 差 8e−17；h 变小后误差上升是差分相消，属预期。

## 6. 结论

1. 自适应环扩展在这个真实翼身半模上按设计收敛：仅 face 238/359 从 0.0 的奇异比扩到第三环，分别达到 0.6945 / 0.7133，全网格最小 0.1141，未放宽阈值、未跨越 TE/接触切边。
2. 翼身半模现在可以真正求解：LU 残差 9.9e−16、GMRES 正向 3.1e−11 / 转置 1.8e−11、伴随恒等式 2.2e−16。
3. 镜像 oracle 在翼身半模上逐位一致（含力/矩/CL/CM 与站位导数），降阶等价性在该离散上成立。
4. 该 418 未知量规模下 GMRES 仍比 LU 慢、内存更高；GMRES 的价值仍是内存（大 N），不是速度。
5. **物理精度仍未验收**：`physical_accuracy_accepted=false`，本轮不算 CLA/CD 的物理验证；CL 0.53123 也不与旧 488 面独立全模（0.52246）或纯机翼半模（0.51209）可比，它们是不同离散。

## 7. 产物

`runs/half_wb_rank2_reject_v1/`（对照失败）、`runs/half_wb_rank6_lu_v1/`、`runs/half_wb_rank6_gmres_v1/`、
`runs/half_wb_rank6_transpose.json`、`runs/half_wb_rank6_mirror_v1.json`、`runs/pr22_peak/{rank2,rank6,rank6gmres,wb_mirror,transpose}/`、
诊断脚本 `runs/pr22_audit_dump.py`、`runs/pr22_depth.py`、`runs/pr22_transpose.py`。
