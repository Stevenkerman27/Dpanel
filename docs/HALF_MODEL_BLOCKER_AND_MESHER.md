# 半模 blocked：诊断、TE 门控修正与镜像网格生成路线

**状态：诊断修复已提交；真实 Gmsh 网格半模尚未解除 blocked。** 基于
`local/panel-jax-20261025` 的 Issue #16 / PR #17 本地回执。研究代码不得替换默认
`panel/solve.py`、`aero.py`，也不改变 `physical_accuracy_accepted=false`。

## 1. 已核实的阻塞

PR #17 本地 JAX 0.10.2 CPU 三个真实算例的 `--half` 都在
`panel/iterative.py:95-101` 的**全顶点（含尾迹节点）镜像双射检查**直接
`ValueError: Need bijective mirrored vertices, including wake nodes`；
没有开始 GMRES，不能称为预条件器或伴随收敛失败。

Issue #16 发现同一类问题：`single_wing` 484 壁面中仅 26 面有精确镜像重心伙伴，
镜像重心最近距离中位约 0.01329 弦长；
`two_bodies_coarse` 616→56、`forward_body` 696→74、
`two_bodies_fine` 1646→68、`aft` 1976→148。
环量站位是严格对称的，但 wall 三角形并非左右同一剖分。
`panel/cad.py:277-299` 当前一次性 `gmsh.model.mesh.generate(2)` 后采集
完整壁面/尾迹两侧三角；几何和尺寸场对称不约束各侧使用同一三角对角线、面片节点 ID。
`panel/build_surface.py:65-71` 对原始网格检查闭合，并不会将左右重新配对。
因此当前阻塞的**根因是前处理离散拓扑**，不是飞行器几何理论上不对称。

报告中的“7 个中心线面元”来自按最近重心寻找自己对应的统计。
它不自动证明存在 7 个虚假 y=0 封盖；必须区分：
(1) 三个顶点都在 y=0 的 planar cap，
(2) 跨过 y=0 的三角，
(3) 仅重心接近/最近匹配自己。这三个情况对修复有不同含义。
新 `panel.audit_half_mesh` 提供相应分项。

## 2. 额外发现并修正的门控错误

旧 `panel/matrix_free.py` 的 `mirror_gate` 将
`pair[upper_te_faces]` 与 `lower_te_faces` 集合比较。
XZ 镜像只变换 y 不变换 z；`panel/build_surface.py:97-100`
恰按 z 的正负定义上/下尾缘面，因此**upper→upper、lower→lower**，
且必须属于**同一条镜像 TE 边**。旧的“64/80 TE 边不匹配”数字会被这个
错误的 bank 规则污染，不能当作独立、可靠的 TE 几何不相容证据。
新代码用成对 bank 检查，另拒绝该试验 reducer 不支持的 self-mirror wall
未知量。此修改不解决全顶点缺少镜像伙伴的问题。

新的 `panel.audit_half_mesh` 还执行：节点镜像距离与双射/对合、壁面与尾迹的
**精确三角 vertex-set 映射**（不只比重心）、法向正确反射、TE 边及上下 bank、
组件标签、站位对称、y=0 全平面三角与跨平面三角分类。
它是只读诊断，不能因 `status=topology_symmetric` 就跳过原有的
`panel/iterative.py` 载荷/伴随/残差验收；矩阵的镜像等变性必须另外检查。

## 3. 真正解除 blocked：改变生成网格的工艺

不要调大容差、把左右独立面元按重心强行对应、平均 AIC，也不要只镜像
现有左/右**完整独立剖分**中的坐标。必须生成**同一个半模拓扑的镜像拷贝**：

1. 在 `panel/cad.py` 的 OCC union / fluid-cut / 接触尾迹支持仍保持真实
   外露翼身公共边界的前提下，建立 **y>=0 正半域**。显式导出
   CAD symmetry-plane 边界/缝 ID；只对半域生成 conforming surface mesh。
   不需要将该临时 symmetry-plane face 作为真实物面边界条件。
2. 选取真正暴露的半侧壁面及 fluid-side wake 三角、TE 两 bank、部件标签；
   完整保留机翼—机身公共交线。对中面根部必须保留由**拓扑索引**定义的 seam。
   对接触切线、中心线穿越 TE、机身跨 TE 的情况需要独立有效性检查。
3. 使用 `(x,y,z)↦(x,-y,z)` 复制严格正 y 节点；所有 y=0 seam 节点
   **共用原有 ID**，不是创建两个重合副本。复制左壁面/尾迹时反转三角绕序，
   例如 `(a,b,c)↦(a',c',b')`；组件面片标签也做相同映射。
   删除的只能是**为半域计算人为生成的非物理 symmetry cap**，不可删去真实外表面。
4. 显式镜像 TE edge 索引及 upper/lower 面片 ID（upper→upper），同步镜像
   站位、`gamma_station_dof`、wake strips/interpolation、
   contact/seam edges、cut-aware 压力邻域。
   `panel/build_surface.py` 的外部面闭合、方向、TE 上下 bank、wake
   边界分类都必须以**新完整网格**重跑。不能把旧面片编号/邻接移植到新网格。
5. 在新完整网格上重跑 `make_plan(..., symmetry=True)` 的严格门控；
   显式比较 `R A E` 与新网格的稠密 `A_{RR}+A_{RL}`、
   全/半状态、`Cp_raw`、逐面/组件/总力矩、`A^\mathsf T` 伴随和
   对称保持的几何随机方向 VJP/FD。由此确认同一个离散系统的等价性。
6. 上述新网格与原来 Gmsh 独立全模**不是同一个离散系统**：
   旧/新物理值差异应按网格收敛和独立基准解释，而不能要求逐面数值完全相同。
   非对称几何、侧滑、非对称操纵面/载荷须回退到**显式全模**。

早期只需要改进**网格生成/固定拓扑**，不必重写 GMRES、矩阵无关核或伴随数学。
半模计算时 `E` 复制左右强度、`R` 选择独立方程，
其转置必须 `Eᵀ Aᵀ Rᵀ`，而几何的镜像张量 VJP 也必须把左右梯度相加，
不能只保留右侧梯度。

## 4. 现在可执行的最小任务

在已拉取本研究分支的 Dpanel 环境：

```bash
python -m pytest -q test/test_half_mesh_audit.py test/test_matrix_free_symmetry_gate.py

python -m panel.audit_half_mesh \
  --surface cases/panel_cut/single_wing.json \
  --out runs/half_audit_single_wing_v1.json
```

第二条即使返回 `status=blocked` 也属于成功的**诊断**；不能当作半模
求解已通过。比较所有真实构型的 `vertices/wall/wake/trailing_edges/stations`
字段，报告具体优先处理的中面 crossing faces 与 seam / wake 残缺情况。

**本分支不改变 CAD/Gmsh 网格，不宣称 half unblocked，也不启动训练。**
半域 OCC 剖分加镜像装配应作为下一独立 PR，在其之前保留全模 LU 或
已经数值验证过的全模矩阵无关方案。

## 5. 从现有固定 fixture 直接复核的分层数据

2026-10-25 直接读取 `local/panel-jax-20261025@430d5646` 上的
`cases/panel_cut/{single_wing,two_bodies_coarse,forward_body}.json`（原始文件分别为
`29f41e0`、`43c43e3`、`9e5d9c6`）。
对 **y=0** 的坐标反射按全节点 `1e-10·max(1,几何跨度)` 进行最近伙伴检查；
“精确三角”必须每个顶点有容差内镜像节点且三角的三个镜像索引构成已存在的整片三角。
以下是**只读几何索引统计**，不是本地 JAX solver 结果：

| fixture | 镜像节点/总节点 | 完整镜像 wall 三角 | wall 跨平面 | 真正全在 y=0 的 wall 三角 | 完整镜像 wake 三角 | 镜像 TE 线段/总 TE | 同时保持上/下 bank 的 TE |
|---|---:|---:|---:|---:|---:|---:|---:|
| single_wing | 399/531 | 26/484 | 17 | 0 | 268/600 | 80/80 | 52/80 |
| two_bodies_coarse | 345/649 | 56/616 | 18 | 0 | 148/708 | 72/72 | 9/72 |
| forward_body | 406/637 | 72/696 | 30 | 0 | 268/600 | 80/80 | 52/80 |

- 旧 Issue #16 的 `forward_body` “74/696”依据重心最近配对；
  此处“72/696”要求**镜像后的三个顶点组成完全相同的三角**，所以两种数字不矛盾。
- 三例都没有三个顶点全部位于 y=0 的 wall cap，
  但确有 17–30 个三角**跨过**中心平面；
  因此修网格时最关键的是先正确切割跨平面面元，而不是盲目删除“7 个 cap”。
- 真实 TE 线段本身都能找到镜像伙伴（80/80 或 72/72），
  **失败的是相邻三角的上/下 bank 对应**，不是 TE 中心线不对称。
  旧 `matrix_free.py` upper→lower 判据造成的“64/80”不能继续当作独立几何事实。
- 尾迹三角同样由全模 Gmsh 分别剖分，必须和壁面一起重建；只镜像壁面还不能解除
  `including wake nodes` 的报错。

新审核器 `panel.audit_half_mesh` 专门输出 `partial_exact_face_pairs`、
`geometry_edges_reflected_even_if_unpaired_mesh` 和 `partial_exact_banks`，
即使整张网格尚不支持合法的全局镜像置换，也可定位哪个层级已配对。
这些“partial”仅作诊断，绝不能替代严格 full symmetry gate。
