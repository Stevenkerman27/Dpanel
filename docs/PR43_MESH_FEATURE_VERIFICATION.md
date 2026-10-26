# PR #43 本地验证：opt-in OCC 特征尺寸回调（真实 Gmsh A/B）

日期：2026-10-26 · 验证者：本地 agent · 状态：`physical_accuracy_accepted=false`

## 1 结论摘要

1. 云端分支按当前 head（`e331788`）在本机可运行：`test/test_mesh_feature_size.py` **15 passed / 0.08 s / exit 0**，真实 Gmsh 半域 CAD 网格四档全部 **exit 0**，无 traceback。
2. `--mesh-features` 在真实 OCC 半域网格上确实生效：椭圆的 LE 带内平均边长从 **0.0779 c → 0.0290 c（−62.8%）**，TE 带从 **0.0689 c → 0.0265 c（−61.5%）**；provenance 记录 `feature_sizing.source_sha256`。
3. **TE 加密会连带重构尾迹**：尾迹面 577 → 1003（+73.9%），因为尾迹必须与 TE 共形；未知量 1045 → 2517。这是需要预算/最小尺寸门控的地方（issue #42 的 B/C 条），不是缺陷，但成本必须显式。
4. 网格质量**不是单调变好**：LE-only 档最小角从 10.22° 掉到 **6.44°**（翼尖附近出现尖角三角形），LE+TE 档回到 11.99°；p99 aspect 从 2.46 降到 2.04–2.05。
5. 物理量刻意不作为接受条件：CL +1.8%/+2.6%，近场 `CD_pressure` **+46.9%/+41.5%**，与已记录的"近场压力阻力随加密单调上升（相消主导）"一致，**不能**读成精度改善。
6. 无 flag 基线：与旧代码产物 `runs/geom_robust/mesh/elliptic_s100` **面元数/拓扑/审计字段一致**（1028 壁面 / 577 尾迹 / 819 节点 / 35 TE 边 / `max_plane_snap` 1.22e-16），但坐标最大差 **0.1436 c** —— 本机 Gmsh 网格本来就不是位可重复的，因此"严格复现"只能到计数与审计级，不能到 SHA/位级。

## 2 环境与输入 SHA

| 项 | 值 |
| --- | --- |
| gmsh | 4.15.2 |
| Python / NumPy | 3.11.17 / 2.4.6 |
| 分支 head | `e331788`（research/occ-half-feature-density） |
| `panel/mesh_feature_size.py` | `4ea6cf2e3d5797731a4a161fb928a69471f8b81931ebc11be4dad8d88943df54` |
| `panel/cad_half.py` | `2d5c13cf0ff3250717821bf09441706437d06e099ecef2eb31bb8d990437b9a1` |
| `docs/MESH_FEATURE_SIZE.md` | `eb0e30a3537bc98dc8b1c5ee53fb88e0672515dcee93df7e64f488f37d61fba6` |
| STEP `elliptic/wing.step` | `dd254707e840ac38b44f8240bcb052a2f1c75f38d75e8d6f424b87f099fb31bf` |
| `elliptic/trailing_edge.json` | `352370c14d4f6f032b8a1e4a885a15812928cf569c39103b41f802aaf8165471` |
| `features_le_only.json` | `710749c9de4950f5a1e91c1008014f7bbd8440776175cdde0423921cd8b45912` |
| `features_le_te.json` | `c6ec991bdf32dc27f7e46f85468db8cc753aa308035efc68fbfd304012553a61` |

特征折线不是从网格猜的：按 CAD 生成器 `runs/geom_robust/make_cases.py` 的 `elliptic_sections` 同一公式导出，`chord(y)=0.05+0.95*sqrt(1-(y/2)^2)`、`le=0.25*(1-chord)`，21 站 y∈[0,2]，与 `cad/elliptic/build.json` 的 `te_first=[0.2875,-2,0]`、root TE=1.0 一致；`target_ratio=0.25`、`radius_chords=0.12`。

## 3 命令

```text
# 测试
set PYTHONPATH=runs\pr43_git
python -m pytest -q test/test_mesh_feature_size.py      # 15 passed

# 网格 A/B（cwd=runs/pr43_git，PYTHONPATH 必须含 audit/runtime/python 才有 gmsh）
python runs/peak_wrap.py panel.cad_half runs/pr43_verify/peak_base_a --out runs/pr43_verify/base_a \
  --wing-cad runs/geom_robust/cad/elliptic/wing.step \
  --trailing-edge runs/geom_robust/cad/elliptic/trailing_edge.json \
  --size 0.1 --far-size 1.5 --radius 5.0 --alpha-deg 5.0 --ncpu 8
# feat_le 追加 --mesh-features runs/pr43_verify/features_le_only.json
# feat_lete 追加 --mesh-features runs/pr43_verify/features_le_te.json

# 求解（LU，压力模板 2 环，上限 6 环）
python runs/peak_wrap.py panel.run_half_native runs/pr43_verify/peak_solve_base_a \
  --surface runs/pr43_verify/base_a/surface.json --out runs/pr43_verify/solve_base_a \
  --mode lu --max-panels 6000 --max-stencil-rings 6
```

## 4 网格度量

| 档 | 壁面 | 尾迹 | 节点 | LE 带面数(占比) | LE 带平均边长/c | TE 带面数(占比) | TE 带平均边长/c | 最小角° | p01 角° | max aspect | p99 aspect | TE 边(与 wake 共边) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_a（无 flag） | 1028 | 577 | 819 | 108 (8.06%) | 0.07794 | 108 (6.52%) | 0.06889 | 10.22 | 23.01 | 4.78 | 2.46 | 35 (35) |
| base_b（重复） | 1028 | 577 | 819 | 108 (8.04%) | 0.07782 | 108 (6.51%) | 0.06889 | 10.22 | 23.01 | 4.78 | 2.46 | 35 (35) |
| feat_le | 1730 | 577 | 1172 | 640 (7.20%) | 0.02899 | 112 (6.49%) | 0.06790 | **6.44** | 28.63 | 5.07 | 2.04 | 35 (35) |
| feat_lete | 2500 | **1003** | 1743 | 653 (7.17%) | 0.02873 | 640 (5.90%) | 0.02648 | 11.99 | 29.13 | 4.79 | 2.05 | **99 (99)** |

LE/TE 带 = 面心或边中点到特征折线距离 ≤ 0.05 c（5% 弦长带）。耗时：6.1 / 6.1 / 9.4 / 14.0 s；峰值 RSS：211.5 / 209.0 / 211.1 / 210.3 MB。

- 三档审计字段全部正常：`max_plane_snap` 1.2246e-16、`symmetry_plane_vertex_count` 34/38/44、`no_artificial_symmetry_wall=true`、`wake_wall_contact_edges=0`（本案例无翼身接触）、`wall_cut_kinds` 全 `trailing_edge`。
- TE 边数 35 → 99：TE 曲线被加密后拆成更多 CAD 段，每段都是一个 cut edge；**99 条全部与尾迹边重合**，共形性保持。

## 5 无 flag 基线对照

| 对象 | 壁面 | 尾迹 | 节点 | TE 边 | `public_wall_positive_mirrored_volumes` |
| --- | --- | --- | --- | --- | --- |
| 旧代码 `geom_robust/mesh/elliptic_s100` | 1028 | 577 | 819 | 35 | 0.21534703 |
| 新代码 `pr43_verify/base_a` | 1028 | 577 | 819 | 35 | 0.21534676 |

计数与审计一致，坐标最大绝对差 **0.1436 c**，两文件规范化 SHA 不同。结论：opt-in 关闭时**没有引入结构性变化**，但本机 Gmsh 网格本来就不是位可重复的（多次运行即使同代码也不同），所以"严格复现"只能在计数/审计层面声明。

## 6 求解读数（LU，不是物理接受条件）

| 档 | 未知量 | CL | `CD_pressure` | `CM_y` | `Cp` min | `Cp` max | 真残差 | 热解 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_a | 1045 | 0.373347 | 0.003291 | 0.001088 | −2.788 | 0.813 | 1.06e-15 | 0.437 |
| feat_le | 1747 | 0.380156 | 0.004834 | 0.003007 | −1.887 | 0.9999 | 9.67e-16 | 0.897 |
| feat_lete | 2517 | 0.383181 | 0.004657 | 0.001724 | −1.850 | 0.9998 | 1.24e-15 | 2.146 |

- 三档压力模板审计：`initial_rank_deficient_faces=[]`、`expanded_faces=[]`（无需扩环），`pressure_neighbor_width=9`。
- CL 单调上升 1.8%/2.6%，`CD_pressure` 上升 46.9%/41.5%，`|Cp_min|` 从 2.79 降到 1.85–1.89。后者与"粗网格在前缘产生更尖的离散吸力峰"一致，但在没有独立参考前**不解释为精度改善**；`CD_pressure` 的上升与已记录的 396→4371 近场阻力单调上升同源（正负相消主导）。

## 7 发现（按重要性）

1. **回调确实只降不升、且作用在 CAD 曲线与面上**：LE 带内边长 −62.8%，带外面元数不变（feat_le 尾迹 577 与 base 相同）。
2. **TE 加密的连带成本没有门控**：尾迹 577→1003、未知量翻 2.4 倍。issue #42 要求"最小尺寸/总面元预算，超限中止而不是默改尺寸"，当前实现没有总预算，只有 1000 段折线上限。建议加 `max_wall_panels`/`max_total_panels` 预算，超限 `ValueError` 中止。
3. **网格质量可能反向**：LE-only 最小角 10.22° → 6.44°。`docs/MESH_FEATURE_SIZE.md` 已声明"只保证请求 size 的数学性质，不保证实际元素质量"，本实测正好验证了这条警告——审计脚本应作为标配，而不是可选。
4. 小改进项：`radius_chords` 与 `target_ratio` 没有交叉校验（radius 很小时回调几乎不生效但会静默通过）；provenance 只记源 SHA 与特征名，未记 `target_ratio/radius_chords` 的数值摘要，回放时无法仅凭 surface.json 复现请求。

## 8 未做 / 不能声称

- 镜像闭合 oracle（`panel.verify_half_mirror` 需要稠密 E，2500 面代价过高）；本报告只用 `surface.json` 审计字段判断对称面与人工 cap。
- 接触割边（椭圆翼无翼身接触）；GCI/渐近性（单 h，且本机网格非位可重复，跨档 face ID 不可比）。
- 曲率驱动、PostView 后验误差、外部 VSPAERO/CFD 真值。
- 任何"物理精度改善"结论。
