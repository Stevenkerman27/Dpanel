# PR #47 本地验证：OCC 三角质量与预算门控

对象 research/mesh-quality-budget-gates @ cf4cb52191a1b8c8fcb1df99e0ec38ae17ddc049（worktree runs/pr47_git，base research/occ-half-feature-density@e331788）。本机 gmsh 4.15.2 / Py 3.11.17 / NumPy 2.4.6 / JAX 0.10.2 / Windows 10.0.26200。

physical_accuracy_accepted=false。全部结论只关于**离散与资源**，不是物理精度验收。

## 1. 单元测试

test/test_mesh_quality_gate.py + test/test_mesh_feature_size.py → **32 passed / 0.13 s / exit 0**（本机真实 JAX/NumPy 环境，非容器）。

## 2. 真实 Gmsh A/B（椭圆翼，STEP + TE 声明，--size 0.1 --far-size 1.5 --radius 5 --alpha-deg 5 --ncpu 8）

| 档 | 壁面 | 尾迹 | 节点 | 壁面最小角° | 相邻面积比 p99 | 最小形状质量 | TE 边/wall/wake 共边 | 模板最小奇异比 | 耗时 | 峰值 RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base（无 flag） | 1026 | 577 | 818 | 10.22 | 2.52 | 0.294 | 35/35/35 | 0.2436 | 5.0-5.6 s | 211.6 MB |
| feat_le | 1740 | 577 | 1177 | **6.44** | 2.42 | 0.161 | 35/35/35 | 0.2329 | 7.4-8.0 s | — |
| feat_lete | 2498 | 1003 | 1742 | 11.99 | 2.41 | 0.352 | 99/99/99 | 0.2420 | 10.7-11.3 s | 209.3 MB |
| feat_lete + 宽松门控 | 2494 | 1003 | 1740 | 11.99 | 2.43 | 0.352 | 99/99/99 | 0.2420 | 10.7 s | — |

base 行无 --mesh-features/门控 → audit.mesh_quality **不存在**（符合「无 flag 不新增输出」），该行数字是把同一份 surface.json 喂给 audit_mesh_quality 直接算出来的。feat_le/feat_lete 即使不加门控也会因为启用特征而附加 audit。

宽松门控 --max-wall-panels 4000 --max-wake-panels 2000 --max-total-panels 6000 --min-wall-angle-deg 4 --max-wall-growth-p99 6 全部通过并写出完整 surface.json。**只给门控、不给特征**（--max-wall-panels 4000 --max-total-panels 6000）也正确触发 audit 并 exit 0。

## 3. 真实拒绝（四种门控各命中一次）

| 门控 | 实际值 | 报错（failure.json.message） |
| --- | --- | --- |
| --min-wall-angle-deg 8（LE-only） | 6.44424° | Wall mesh angle gate failed: min_angle_deg=6.44424 < 8 |
| --max-wall-panels 2200（LE+TE） | 2496 壁面 | Mesh budget exceeded: wall_panels=2496 > max_wall_panels=2200 |
| --max-wake-panels 900（LE+TE） | 969 尾迹 | Mesh budget exceeded: wake_panels=969 > max_wake_panels=900 |
| --max-wall-growth-p99 1.5（LE-only） | 2.40375 | Wall mesh growth gate failed: p99_neighbor_area_ratio=2.40375 > 1.5 |

四条都 exit 1、写 failure.json（含 stage=gmsh_positive_half_surface_meshing、exception=ValueError、完整 traceback、elapsed_seconds），**不写 surface.json**，不覆盖已有目录。云端预测的「LE-only 最小角 6.44」与实测**逐位一致**。

## 4. 无 flag 等价性

对比 PR #43 的 runs/pr43_verify/base_a（同 STEP、同参数、旧 commit e331788）：

- 顶层键集合**完全相同**（19 个）；audit 键集合**完全相同**（11 个，无 mesh_quality）；provenance 深度相等（prov equal: True）。
- 差异只有 half_wall_panels（1026 vs 1028）与 public_wall_positive_mirrored_volumes（0.21535587 vs 0.21534676）——即**本机 Gmsh 网格非位可重复**，不是新代码改变行为。

## 5. 四条发现（建议进 PR）

1. **尾迹三角本身就是极薄片，且与特征加密无关**：直接审计 base（无 flag）得到 wake 最小角 **0.05°**、最小形状质量 **0.002**、面积 60.310；LE-only 与 LE+TE 的 wake 面积也是 60.310（尾迹片未变），最小角 0.05°/1.10°。所以 audit.wake.min_angle_deg 由**尾迹条带构造**决定，不能用来评价壁面加密质量。PR 目前没有 wake 角度门控（正确）；若以后要加，必须给尾迹专用下限（本 Geometry 上 ~0.5-1° 才有意义），不能复用 --min-wall-angle-deg。
2. **门控参数校验发生在网格生成之后**：--max-wall-panels 0 与 --min-wall-angle-deg 60 都是**先跑完整 5.6 s 网格**、写出 te_check_*.json，然后才在 audit_mesh_quality 里抛 max_wall_panels must be a positive integer / min_wall_angle_deg must be finite and within bounds。建议在 main() 解析后、gmsh.initialize() 之前加一次纯参数校验（几行，零成本），否则 2498 面板档要白跑 11 s 才发现参数写错。
3. **--max-wall-growth-p99 在本 Geometry 上区分度很低**：base 2.52 → LE-only 2.42 → LE+TE 2.41，几乎不动；真正拦住过度聚拢的是**角度门控**（6.44 < 8）。可以保留该门控（成本为零），但文档不宜把它写成质量主门控，且默认值不要给太紧（>2.5 才不会被正常网格误伤）。
4. **门控不改变内存/时间成本**：base 211.6 MB / feat_lete 209.3 MB，耗时 5.0 s / 10.9 s（含审计），与 PR #43 未加门控时同量级。符合 PR 的「事后门控不约束 mesher 自身内存」声明；反过来也说明它无法防止 mesher OOM。

## 6. 复现命令

$env:PYTHONPATH='D:\3D\Projects\panel\runs\pr47_git;D:\3D\Projects\panel\audit\runtime\python'
$env:TMP=$env:TEMP=$env:TMPDIR='D:\3D\Projects\panel\runs\pipwork'
Set-Location D:\3D\Projects\panel\runs\pr47_git
& ..\..\.conda_env\python.exe -m pytest -q test/test_mesh_quality_gate.py test/test_mesh_feature_size.py -p no:cacheprovider
& ..\..\.conda_env\python.exe -m panel.cad_half --wing-cad ..\geom_robust\cad\elliptic\wing.step --trailing-edge ..\geom_robust\cad\elliptic\trailing_edge.json --size 0.1 --far-size 1.5 --radius 5 --alpha-deg 5 --ncpu 8 --out ..\pr47_verify\feat_lete --mesh-features ..\pr43_verify\features_le_te.json

（Gmsh 只在 audit/runtime/python 里，必须让 PYTHONPATH 同时含 worktree 与它。）

证据：runs/pr47_verify/{compare.json, 各 case 的 surface.json/manifest.json/failure.json}；汇总脚本 runs/pr47_verify/summarize.py。

## 7. 未做

- 未跑重复 Gmsh 种子的统计（本机网格非位可重复，已用三档实测差异说明，但没做 seed 扫描）。
- 未做 5k-12k 壁面档：稠密 LU 与 mesher 内存上限都没探。
- 未评估门控对求解精度的影响（门控只拒绝，不改变网格；同网格的 CL/CD 差异见 PR #43 验证报告）。
- 未接触生产默认路径、压力重构、核、尾迹模型、Gamma 或训练。