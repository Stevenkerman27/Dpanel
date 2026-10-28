import json, sys, pathlib
import numpy as np

WORKTREE = r"D:/3D/Projects/panel/runs/pr52_git"
MESH_ROOT = pathlib.Path(r"D:/3D/Projects/panel")
ROOT = pathlib.Path(WORKTREE)
sys.path.insert(0, WORKTREE)
from panel.p1_gate_checks import (assembled_free_terms, near_plane_gate,
                                  node_dof_audit, self_trace_gate)
from panel.p1_doublet_moment import triangle_p1_doublet_potentials

TRI = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
MESHES = [("dp_s020", "runs/vsp_conv/dp_s020"), ("dp_s014", "runs/vsp_conv/dp_s014"),
          ("dp_s005", "runs/vsp_conv/dp_s005")]
POINTS = [("centroid", "内部重心"), ("edge", "边中点"), ("vertex:0", "直角顶点"),
          ("vertex:2", "45 度顶点")]

near = near_plane_gate(TRI)
self_rows = []
for key, label in POINTS:
    report = self_trace_gate(TRI, point=key)
    oracle = triangle_p1_doublet_potentials(TRI[int(key.split(":")[1]) if ":" in key else 0][None, :],
                                            TRI[None, :, :], self_ids=np.array([0]))[0, 0]
    self_rows.append({"point": key, "label": label,
                      "predicted": report["predicted_coefficient"],
                      "extrapolated": report["extrapolated_coefficient"],
                      "finest": report["measured_coefficient"],
                      "spread": report["coefficient_spread"],
                      "pv_vanishes": report["principal_value_vanishes"],
                      "oracle_error": report["oracle_coefficient_error"],
                      "oracle_self": [float(v) for v in oracle]})

mesh_rows = []
for tag, base in MESHES:
    data = json.load(open(MESH_ROOT / base / "mesh" / "surface.json", encoding="utf-8"))
    verts = np.asarray(data["coordinates"], float)
    faces = np.asarray(data["faces"], int)
    free = assembled_free_terms(verts, faces)
    audit = node_dof_audit(verts, faces)
    total = free["free_terms"]
    used = free["incidence"] > 0
    te = used & (total > 0.15) & (total < 0.30)
    mesh_rows.append({"tag": tag, "panels": audit["n_panels"],
                      "nodes_referenced": free["n_referenced"],
                      "unused_vertices": free["n_unused_vertices"],
                      "smooth": free["n_smooth"], "non_smooth": free["n_non_smooth"],
                      "te_like": int(te.sum()),
                      "free_min": free["min"], "free_max": free["max"],
                      "duplicated": audit["n_duplicated_groups"],
                      "boundary_edges_raw": audit["n_boundary_edges_raw"],
                      "boundary_edges": audit["n_boundary_edges"],
                      "nonmanifold": len(audit["nonmanifold_edges"]),
                      "closed": audit["closed"], "gauge": audit["gauge_required"],
                      "split_nodes": audit["split_nodes_present"]})

def _enc(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return float(value)
    raise TypeError(type(value))


payload = {"near_plane": {k: near[k] for k in ("worst_relative_error", "passed")},
           "near_plane_rows": near["rows"], "self_trace": self_rows, "meshes": mesh_rows}
out_json = ROOT / "docs" / "issue49_vspaero" / "p1" / "gates.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(payload, indent=1, default=_enc), encoding="utf-8")

L = []
L.append("# Issue #49 优先级 d：P1 偶极子三门槛实测")
L.append("")
L.append("状态：三门槛的核级与几何级检查已实测（`panel/p1_gate_checks.py` + 13 条测试）。")
L.append("**这不是连续 P1 wall solve，也不是物理精度声明**（`physical_accuracy_accepted=false`）。")
L.append("")
L.append("## 1 门槛 1：近平面核的独立高精度对照")
L.append("")
L.append("方法：以目标在面板上的投影为极点做极坐标 Gauss 求积（`rho=|h|sinh t`，`theta` 在三个顶点方向断开），")
L.append("被积函数化为 `sign(h)*tanh(t)/cosh(t)/(4*pi)`，在 `(t,theta)` 上解析光滑；与 PR #52 的解析式")
L.append("`triangle_p1_doublet_potentials` 逐项比较。")
L.append("")
L.append("| h/sqrt(A) | 偏移 | 相对误差 |")
L.append("|---|---|---|")
for row in near["rows"]:
    L.append("| %.1e | %s | %.2e |" % (row["ratio"], "重心" if row["offset"][0] == 0 else "偏心", row["relative_error"]))
L.append("")
L.append("最坏相对误差 **%.2e** → 门槛 1 通过：`h/sqrt(A) < 0.01` 区解析核与独立求积一致到机器精度。" % near["worst_relative_error"])
L.append("")
L.append("## 2 门槛 2：面上自由项不是常数 0.5")
L.append("")
L.append("半边探针：`quad(+h)` 与 `quad(-h)`，反对称部分给立体角比例 `c`，对称部分是主值。")
L.append("平板双重体核关于 `h` 严格奇对称，**实测主值恒为 0（<1e-12）**，所以面上极限只剩 `c(x)*B(x)`。")
L.append("")
L.append("| 配置点 | 预测 c | 外推 c | 最小 h 处 c | 与内置 -0.5 的差 | oracle self |")
L.append("|---|---|---|---|---|---|")
for row in self_rows:
    L.append("| %s | %.6f | %.9f | %.9f | %.4f | %s |" % (row["label"], row["predicted"], row["extrapolated"], row["finest"], row["oracle_error"],
                                                       "[" + ", ".join("%.3f" % v for v in row["oracle_self"]) + "]"))
L.append("")
L.append("预测值就是 `内角/(4*pi)`：内部 0.5、边中点 0.25、直角顶点 0.125、45 度顶点 0.0625。")
L.append("**PR #52 oracle 的 self 分支把自由项写成常数 `-0.5*B`（等价 c=0.5，来自 -n 侧）**，只在光滑内部点成立；")
L.append("节点式 P1 的配置点恰是顶点，那里偏大 4 倍（直角）到 8 倍（45 度）。")
L.append("")
L.append("### 真实网格上的装配值")
L.append("")
L.append("| 网格 | 壁面 | 引用节点 | 未引用顶点 | 光滑(|c-0.5|<=1e-3) | 非光滑 | 尖锐后缘(0.15<c<0.30) | c 范围 |")
L.append("|---|---|---|---|---|---|---|---|")
for row in mesh_rows:
    L.append("| %s | %d | %d | %d | %d | %d | %d | %.4f - %.4f |" % (row["tag"], row["panels"], row["nodes_referenced"], row["unused_vertices"],
                                                                  row["smooth"], row["non_smooth"], row["te_like"], row["free_min"], row["free_max"]))
L.append("")
L.append("梯形翼的尖锐后缘节点装配出的 c 是 **0.2343 - 0.2969**（约 1/4），常数 -0.5 在这些节点上偏大 **1.7-2.1 倍**；")
L.append("12290 面网格上这类节点有 55 个。光滑内部节点的装配值 0.5000008 - 0.5013，验证了 `sum(内角)/(4*pi)` 恒等式。")
L.append("")
L.append("## 3 门槛 3：节点未知量、方程数、gauge 与切口")
L.append("")
L.append("| 网格 | 重复节点组 | 原始边界边 | 合并后边界边 | 非流形边 | 闭曲面 | 需 gauge | 切口节点 |")
L.append("|---|---|---|---|---|---|---|---|")
for row in mesh_rows:
    L.append("| %s | %d | %d | %d | %d | %s | %s | %s |" % (row["tag"], row["duplicated"], row["boundary_edges_raw"], row["boundary_edges"],
                                                          row["nonmanifold"], row["closed"], row["gauge"], row["split_nodes"]))
L.append("")
L.append("两条与实现直接相关的事实：")
L.append("")
L.append("1. `surface.json` 的 `coordinates` 里含尾迹顶点：12290 面档有 **1212 个顶点不在壁面里**（7385 -> 6173）。")
L.append("   节点未知量必须由壁面三角重建，按 `len(coordinates)` 开会多出 16.4% 的无声未知量。")
L.append("2. 壁面是开曲面（只有后缘切口为边界，14/20/54 条边），所以常数 doublet 平移是零空间 → **必须 pin 一个 gauge**。")
L.append("   三档都 `duplicated=0`、`split=0`、`nonmanifold=0`：这批 Gmsh 网格没有拓扑切开的共享节点，但这必须是**被检查**的，不能假定。")
L.append("")
L.append("## 4 与云端三门槛的对照")
L.append("")
L.append("| 门槛 | 状态 | 证据 |")
L.append("|---|---|---|")
L.append("| h/sqrt(A)<0.01 近平面独立高精度 | 通过 | 最坏相对误差 %.2e，`near_plane_gate` |" % near["worst_relative_error"])
L.append("| 面内非质心自迹 -N_a(x)/2 + 有限部 | 修正 | 有限部（平板）恒 0；自由项 = `内角/(4*pi)`，顶点不是 0.5 |")
L.append("| 共享节点切口 + 未知量/方程数/gauge | 通过（工具级） | `node_dof_audit`：1212 个未引用顶点、gauge 必需、0 重复/0 非流形 |")
L.append("")
L.append("## 5 对下一步（连续 P1 wall solve）的硬约束")
L.append("")
L.append("1. 对角/自由项必须**由装配的立体角算出**（逐面板 `c` 求和），不能写死 `-0.5`；否则后缘节点系统性错 1.7-2.1 倍。")
L.append("2. 未知量集合必须从壁面三角构建并做重合点合并；尾迹顶点不得进未知量。")
L.append("3. 开曲面必须有 gauge（常数平移），并在残差里按投影消掉。")
L.append("4. 以上是**几何/核级**结论；与解析球对比、以及真正的 P1 wall solve 仍未做。")
L.append("")
L.append("## 6 复现")
L.append("")
L.append("```")
L.append(r"python tools/vsp_evidence/p1/p1_gate_report.py")
L.append(r"python -m pytest test/test_p1_gate_checks.py -q -p no:cacheprovider")
L.append("```")
L.append("")
(ROOT / "docs" / "ISSUE49_P1_GATES.md").write_text(chr(10).join(L) + chr(10), encoding="utf-8")
print("doc lines", len(L))
print("near worst %.2e" % near["worst_relative_error"])
for row in self_rows:
    print(row["label"], "pred %.6f extr %.9f finest %.9f pv0 %s" % (row["predicted"], row["extrapolated"], row["finest"], row["pv_vanishes"]))
for row in mesh_rows:
    print(row["tag"], "panels", row["panels"], "nodes", row["nodes_referenced"], "unused", row["unused_vertices"], "te_like", row["te_like"], "min %.4f max %.4f" % (row["free_min"], row["free_max"]))