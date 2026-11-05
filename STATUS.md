# 项目状态

更新：2026-11-05。本文件是根目录唯一的状态入口；2026-10-26 的旧状态文档已移到 `docs/archive/root/`。

## 一句话

JAX 可微气动面元求解器（解析源/偶极子 + 真实流体侧接触尾迹），另有全势/P2 对照线；几何与离散导数在声明范围内通过，**物理精度尚未验收**（`physical_accuracy_accepted=false`）。

## 代码基线

- `master` = `517e20e`。2026-11-04 通过 PR #109 把研发分支 `work/root-2026-11-03` 整条线 squash 并入（仓库禁止 merge commit），再通过 PR #110 合入 P1 自由尾迹前驱图审计。本地主工作树跟踪 `work/root-2026-11-03`，已与 master 同点。
- 合并内容：docs 清理（520 文件/28.7 MB → 460 文件/17.6 MB，废弃证据进 `docs/archive/`）、`tools/web` 浏览器工作台、`panel/` 精简（删 20+ 个旧模块）、上反角支持、P1 G1/G2 合并内核。
- 分支现状：`master`（唯一基线）+ `work/root-2026-11-03`（唯一开发分支，内容与 master 相同，后续从它继续）。临时分支（`pr107-review`、`dihedral-solver`、`merge/dev-to-master` 等）已删除。
- 测试基线（2026-11-04，`python -m pytest -q test -p no:cacheprovider`）：合并后 master 实测 `16 failed, 796 passed, 21 skipped`（184.05 s，`runs/scratch/master_verbose.out`）；合并树当时 `16 failed, 793 passed, 22 skipped`（177.97 s），dev 分支单独跑同为 `16 failed, 793 passed, 22 skipped` 且失败集合逐条相同 → 合并本身没有引入新失败。
- 环境 flake（已知，未修）：全量跑两次都在第 ~36 个测试处崩 `Windows fatal exception: access violation`（exit 139，栈在 `jax/_src/interpreters/mlir.py` 的 `_create_ir_context`；更早的 master 上同一位置换成 `scipy.linalg.lstsq`）。同一命令第三次跑完，结果即上面的 796 passed。复现命令：`python -m pytest -v test -p no:cacheprovider > runs/scratch/master_verbose.out 2>&1`。
- 16 个已知失败集中在：`test_torch_panel_bridge`（3）、`test_iterative_panel`（3）、`test_half_native_panel`（2）、`test_cad_half_groups_wake_sheet`（2）、CLI 快照类（`test_p1_closed_body`、`test_pressure_recovery_half_integration`、`test_pressure_region_budget`、`test_wake_velocity_methods`）、`test_half_mirror_oracle`、`test_differentiable_panel_fields`、`test_torch_half_lifting`。均为合并前既有失败，未修。
- 浏览器工作台（改参数→建体→独立生成网格→独立求解，`python -m tools.web.server --port 8811`）见 [tools/web/README.md](tools/web/README.md)。2026-11-05 验证：`node --check tools/web/static/app.js`、`C:/Users/zyx20/anaconda3/envs/myml/python.exe -m py_compile tools/web/server.py tools/web/pipeline.py`；本地 API 顺序 `POST /api/build` → `POST /api/mesh` → `GET /api/job` → `GET /api/mesh` 实测默认网格 204 壁面 + 242 尾迹面板，未先生成网格及尺寸不匹配的求解请求均返回 HTTP 409。完整求解未执行。它只包装现有求解器；自由尾迹塌陷缺陷在工作台里以黄色警告呈现。
- Web 大网格访问冲突已定位（2026-11-05）：9950X 上 conda OpenBLAS 0.3.34 的 Ryzen 自动调参把 Haswell 的 Q 从 Windows 默认 128 覆盖成 512，汇编临时缓冲仍只按 K=128 预留；实际 4612 阶机翼捕获 K=256 时写至 `StackBase`，与 14:20/14:21 原始 WER 的 `dgemm_kernel_HASWELL+0x16e` 同一复制循环。`python -m tools.blas_probe --case {gemm,lu,jax} --size {1024,4612,4612} --out ...` 三类独立矩阵各 3/3 崩溃；同一 Haswell 只在诊断进程改 Q=128 后各 3/3 正确。探针、命令、寄存器证据索引见 [tools/web/README.md](tools/web/README.md) 的根因节。单一 myml 环境：Python 3.13.14 / NumPy 2.4.6 / SciPy 1.18.0 / JAX 0.12.0.1；未安装第二环境。按用户要求撤回生产入口自动绕行；当前阻塞是运行库调参与内核容量不一致，下一步为正式修复/独立验收运行库。仅记录 Sandybridge 诊断对照原网格可完成（CL 0.5914475755、代数残差 2.83e-15），物理精度未验收。分组求解相关四文件测试实测 `41 passed in 9.29s`（`OPENBLAS_CORETYPE=Sandybridge OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q test/test_shed_group_blocks.py test/test_shed_groups_contract.py test/test_shed_group_mass_identity.py test/test_run_half_groups_entry.py -p no:cacheprovider`）。
- P1 单翼与多组已合并成一个内核（2026-11-04）：`p1_lifting_g1.solve` 现在是 `solve_multi(combine_wings([wing]))` 的适配器，旧的单翼 torch 核 `p1_solve_torch.solve_single_torch` 已删除。合并前实测单组与旧路径 CL 差 3e-14、Γ 差 5e-14。
- 上反角支持（`dihedral_deg` ±20°，2026-11-04 随 #109 进 master）：后缘不再要求 z=0，尾迹是后缘沿 x 拉伸出的 +z 图面；求解层 5 处“水平尾迹”断言放宽为“+z 图面”，`panel/kernel.py` 的线性环量项补了 `n_y` 修正（旧的 +z 假设在 20° 时把线性项放大 13 倍，216 个旋转算例对求积的最大误差 3.3e-6）。实测 0/5/10/20°：CL +0.428/+0.457/+0.461/+0.439，CD 0.0095/0.0102/0.0120/0.0102；平面路径逐位不变（新旧差 1.4e-16）。物理精度未验收。
- P1 自由尾迹审计（PR #110，`panel/p1_steady_free_wake.py:wake_material_graph_audit`）：每次迭代前记录前驱图（直连边中位数/最大值、最长链、环与 x 回折计数），纯诊断、不改物理。它回答“26 弦直连边 vs 28 弦累计路径”这类问题；26 弦直连边的根因仍未定。
- B3 的 DPanel/VSPAERO 对照（48 项判定，4 项失败）见 [docs/ISSUE92_B3_VSPAERO_PROTOCOL.md](docs/ISSUE92_B3_VSPAERO_PROTOCOL.md)。自由尾迹的原生崩溃（`combine_wings` 后 0xC0000005，根因是 torch 环境 numpy 2.5.2 的大 dgemm 缺陷）已修，四个算例都能跑完 60 次迭代；但只有 rect 收敛，三个 CAD 翼的载荷在松弛中塌陷约四分之三，属未解决缺陷（协议 §13）。
- 已退役的研究代码（P1 配点/Galerkin 变体等）见 [docs/RETIRED_RESEARCH.md](docs/RETIRED_RESEARCH.md)。

## 文档入口

| 想知道 | 看 |
| --- | --- |
| 协作、分支、PR 与验收规范 | [AGENTS.md](AGENTS.md) |
| 现役文档索引与证据包 | [docs/README.md](docs/README.md) |
| 历史报告（PR/issue/notes/evidence） | [docs/archive/README.md](docs/archive/README.md) |
| 案例怎么跑 | [docs/README.md](docs/README.md) 的索引，每篇开头有复现命令 |
| 验收口径（代数 vs 物理） | [docs/WORKFLOW.md](docs/WORKFLOW.md) |
| 测试现状与已知失败 | [docs/TEST_STATUS.md](docs/TEST_STATUS.md) |

## 合并记录（2026-11-04）

- PR #109 `merge: docs cleanup + tools/web workbench + panel slimming onto master`：squash 合并，44 提交、630 文件；合并树测试与 dev 分支完全一致（见上）。仓库设置禁止 merge commit，只能用 squash。
- PR #110 `feat(panel): audit P1 free-wake predecessor graph per iteration`：取代 PR #107（后者有两个缺陷：非单调边按遍历起点重复计数、测试桩缺 `wake_material_graph_audit` 导致 6 个 mock 测试 ImportError）。验证：基线 48 passed，补丁 50 passed。
- PR #99（docs 清理 → research 分支）内容已由 #109 进 master；PR #98（issue92 research 线）仍开放，等决定。
- 沙箱限制：`git push`、`git fetch`、GitHub API 都需要 `danger-full-access` 提权（沙箱禁止管道 stdio，凭据助手取不到 token）；`git merge` 在本仓库会尝试 stash 而失败，需要合并时用 `git merge-tree --write-tree` + `git commit-tree` 造合并提交。

## 旧状态文档的处理

2026-10-26 的 `README`、`DEVELOPMENT`、`PAUSE_HANDOFF`、`SOLVER_ROUTE`、`WBCAERO_STUDY`、`cloud_agent_prompt`
都写着“本地/云端暂停”，与当前 [AGENTS.md](AGENTS.md) 的协作规范冲突，已归档到 `docs/archive/root/`。
其中仍然有效的技术判断（路线选择、全势对照、WBCAero 研究边界）保留在归档原文里，不再作为现行状态。
