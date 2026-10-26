# 文档索引（每个模块只有一份入口文档）

本目录存放跨模块的设计文档、云端协作的验收报告与证据快照。**每个模块只有一个入口文档**，
下面列出「模块 → 入口文档 → 附属文档」，其余文档都是这些文档的支撑材料或历史快照。

## 模块入口文档

| 模块 | 入口文档 | 说明 |
| --- | --- | --- |
| `panel/`（JAX 可微面元求解器） | [`panel/README.md`](../panel/README.md) | 唯一入口：求解链路、验证矩阵、可微性、内存分块、矩阵无关与半模 |
| `torch_panel/`（GRASS 可微 Torch 面元内核） | [`torch_panel/README.md`](../torch_panel/README.md) | 迁移来源、坐标系约定、测试入口 |
| `monitor/`（GitHub 监听） | [`monitor/README.md`](../monitor/README.md) | 用法、状态文件、发评论脚本、沙箱注意事项 |
| `test/`（后端探针与迁移测试） | [`test/README.md`](../test/README.md) | 各测试文件用途与运行环境 |
| `core/`（旧 P2 核心） | [`core/README.md`](../core/README.md) | 与 `core/ACCEPTANCE.md`、`core/RESULTS.md` 配套 |
| `lifting3d/`（三维升力面） | [`lifting3d/README.md`](../lifting3d/README.md) | 协议/结果文档按主题保留在本目录 |
| `potential/`（二维势流/跨声速） | [`potential/README.md`](../potential/README.md) | 与 `potential/PROTOCOL.md`、`RESULTS.md` 配套 |
| `volume/`（体积网格与自适应） | [`volume/README.md`](../volume/README.md) | 与 `volume/MESH.md` 等协议文档配套 |
| `airfoil2d/`（翼型二维） | [`airfoil2d/README.md`](../airfoil2d/README.md) | 与 `airfoil2d/PROTOCOL.md`、`RESULTS.md` 配套 |
| `audit/`（后端逆向与验收） | [`audit/REPORT.md`](../audit/REPORT.md) | 外部工具链与上游调研，不入库；`audit/wbcaero_rea` 是 WBCAero 实现参考，**永久保留不归档** |
| 仓库总体 | [`../README.md`](../README.md) | 顶层定位与目录说明 |

## 本目录文档

### 设计/方法论

| 文档 | 内容 |
| --- | --- |
| `PANEL_ROBUSTNESS.md` | 面元求解器鲁棒性设计与块系统、尾缘闭合、面积积分误差 |
| `WINGBODY_PANEL_LOOP.md` | 翼身组合回路：几何、接触、压强重构 |
| `KERNEL_MEMORY_BLOCKING.md` | 核函数分块 + checkpoint 的内存/时间权衡 |
| `ERROR_SEPARATION.md` | 误差分离实验：几何/离散/势重构/尾缘四类误差预算 |
| `STAGE1_SOLVER_EFFICIENCY.md` | 稠密 AIC 上 LU/GMRES/转置求解的基准与方法 |
| `TORCH_PANEL_EXTRACTION.md` | GRASS PyTorch 面元内核的提取说明 |
| `HALF_MODEL_SCALING.md` | 半模求解随面元数（396/805/1205）的 LU vs 预条件 GMRES 标度与载荷敏感性 |
| `MESH_CONVERGENCE.md` | 网格收敛研究（396–4371 壁面）：$C_L$ 与 Trefftz $C_D$ 收敛到 5%，近场 $C_D$/$C_{M,y}$ 不收敛的诊断 |
| `PAUSE_HANDOFF_2026-10-24.md` | 2026-10-24 暂停交接（历史快照） |

### 云端协作验收报告（按 PR/issue 编号）

| 文档 | 对应线程 |
| --- | --- |
| `PR8_AIC_SYMMETRY.md` | PR #8 / issue #4：VortexAD AIC 镜像不对称归因 |
| `PR10_PRESSURE_RECONSTRUCTION.md` | PR #10 / issue #9：压强重构模板对照 |
| `PR13_VERTEX_VJP_LOCAL.md` | PR #13 / issue #11：顶点几何 VJP |
| `PR15_TORCH_BRIDGE_LOCAL.md` | PR #15 / issue #14：Torch↔JAX 桥 |
| `PR17_ITERATIVE_LOCAL.md` | PR #17：矩阵无关 GMRES（含 JAX 0.10 兼容路径） |
| `PR21_HALF_NATIVE_LOCAL.md` | PR #21：半模原生镜像算子 |
| `PR22_HALF_CAD_LOCAL.md` | PR #22：OCC 正半域网格与半模求解 |
| `PR22_MIRROR_ORACLE_LOCAL.md` | PR #22 第二轮：精确镜像 oracle 与真翼身 CAD |
| `PR22_ADAPTIVE_STENCIL_LOCAL.md` | PR #22 第三轮：自适应压力模板环扩展（失秩面 238/359） |
| `PR23_TORCH_HALF_LOCAL.md` | PR #23：原生 Torch 半模提升面元 LU（CPU + CUDA 实测） |
| `PR24_TORCH_CHECKPOINT_LOCAL.md` | PR #24：Torch 解析核可选 rematerialization 的 CUDA 显存 A/B（5655→438 MB @1988 未知量） |
| `ISSUE16_MATRIX_FREE.md` | issue #16：矩阵无关算子、近场预条件、严格镜像归约 |
| `ISSUE18_INDEPENDENT_SYMMETRY_AUDIT.md` | issue #18：独立对称性审计（伪配对问题） |

### 证据快照

- `panel_robustness_evidence/`、`panel_robustness_evidence_v2/`：鲁棒性矩阵的 summary 与几何图。
- `wingbody_panel_evidence/`：翼身回路的 summary、后端源码哈希与结果图。

## 约定

1. 新增模块先在模块目录放 `README.md` 作为唯一入口，再在本索引登记。
2. 验收报告只写在本目录（`docs/`），**不再往 `runs/` 里放副本**；`runs/` 只放可复现的运行产物。
3. 被取代的文档/脚本移入 `archive/`（见 `archive/README.md`），不删除证据。
4. 临时文件（安装日志、pip 临时目录、一次性调试脚本）一律不进仓库，见 `.gitignore`。
