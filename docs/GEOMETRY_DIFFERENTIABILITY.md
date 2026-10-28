# 固定拓扑几何可微：Dpanel → GRASS-plane 设计变量 VJP

状态：**接口设计 + 最小 JAX 几何 VJP 数值证明**；尚未改动主求解器或完成翼身局部载荷的顶点 VJP。

## 0. 与已有工作的关系

基础为 `local/panel-jax-20261025`（`a169ca64956ccece7d01aa7c9347c53ac678d30c`），已有 37/37 主检查、10/10 拓扑检查、180/180 AD/FD 标量比较。那些导数来自 `panel/solve.py:289` 的 `solve_model(parameters)`，其输入固定为 `[alpha_deg, central_thickness, span_factor]`，其中 `coordinates=jnp.asarray(xyz)` 的原始 `xyz` 在图外由 NumPy 读取。它证明三维翼身网格的**特定连续变形**可微，不等于任意合法顶点坐标的统一 VJP。

CAD/网格入口 `panel/cad.py`、`panel/build_surface.py` 的 Gmsh/OCC 布尔和重网格构造是离散、非 JAX 的预处理；不得声称它们具有端到端自动微分。

当前 `physical_accuracy_accepted=false`。最细测试网格压力 CL=0.319645、环量 CL=0.339294，差约 -5.8%；将导数算准确并不自动修复物理误差。

## 1. 目标接口（下一 PR 实现，不在此 PR 伪造）

```python
# Static, discrete, non-differentiable data created once per topology family.
topology = FrozenTopology(
    faces, wake_faces, cut_edges, upper_te_faces, lower_te_faces,
    component_ids, seam_faces, gamma_station_dof,
    fixed_neighbor_indices,
)
# Values are JAX arrays; *all* quantities depending continuously on vertices
# must stay inside the differentiation graph.
outputs = solve_fields(
    vertices_xyz,              # [V,3], continuous, float64
    alpha_deg, rho, speed,      # continuous operating point
    topology,                  # fixed discrete state / indexes
    gamma_station_geometry,   # geometric parameterization of TE/wake
    reference_area, chord, reference_point,
)
# outputs include raw Cp, panel centers/areas/normals,
# panel_forces [F,3], panel_moments [F,3], total/component loads, diagnostics.
```

函数必须可直接通过 `jax.jit` 与 `jax.vjp` 跟踪。对 GRASS 的单个加权目标不需要构造 `(3F) x (3V)` 完整雅可比：

```python
forces, pullback = jax.vjp(
    lambda X: solve_fields(X, ...).panel_forces,
    vertices_xyz,
)
(dJ_dvertices,) = pullback(upstream_panel_force_seed)
```

用户的 GRASS 模型提供 `X(theta)` 后，利用链式法则把 `dJ/dX` 送回生成器，获得 `dJ/dtheta`。对包含局部结构载荷、CG 位移和机身压力的损失，种子应覆盖完整三维 `panel_forces` 和 `panel_moments`；只有 CL 梯度不能替代这一门槛。

## 2. 保留固定拓扑，但不能冻结错误的几何量

`panel/solve.py` 当前在计算图外固定的合法离散对象：

- faces/wake_faces、cut_edges、上/下 TE 面元 ID、component IDs、gamma_station_dof、cut-aware neighbor_indices（在稳定构型族内）；
- 选定的局部模板覆盖范围/环数，仅在该拓扑和几何合法范围内有效；变形后必须重新检查模板秩与网格有效性。

下列量**不能**冻结为初始几何的 NumPy 数值，需要在可微函数里从新节点动态计算：

1. 真实面元中心、面积、外法向、`sigma=-U.n`、源/偶极子相互作用和固定尾迹影响；
2. 表面局部切向基与 `delta` 切向差、压力重构、原始 Cp、面力作用线；
3. 真实 TE 积分边长、弱 Kutta 表面积分、接触几何条件；
4. 随展向变形移动的 circulation stations、wake linear hat 及 slope。如只对 x/z 微变且 y 站位固定，允许部分常数；**允许任意 y 节点位移时，必须重新定义相容的参数化基**；
5. 参考面积、参考弦、力矩中心及动压是设计量还是固定任务数据，应在接口类型层面显式选择，不可混淆归一化或单位。

尤其当机身跨越后缘、接触组件数发生变化或旧 `gamma_station_dof` 不再代表实际物理约束时，该形变不属于同一个可微拓扑族，必须拒绝并切换离散分支。不能通过 JAX grad 对 CAD 布尔状态求出有效梯度。

## 3. 线性求解及隐式伴随

固定拓扑边界问题可写为

```text
R(u,X) = A(X) u - b(X) = 0
J(X) = objective(X,u(X))
A(X)^T lambda = (d objective / du)^T
dJ/dX = partial J / partial X + lambda^T (db/dX - (dA/dX)u)
```

第一个工程阶段先让 JAX 对 `jnp.linalg.solve(A,b)` 做 AD，证实几何 VJP 的正确性；矩阵块、形变和重构若含 NumPy 转型/截断就不允许放进求导路径。性能足够之后再评估 `jax.lax.custom_linear_solve`、定制 `custom_vjp`、块算子与一次正向/一次转置伴随实现，不应把每个设计变量作为一个独立伴随方程。

## 4. 当前这次 PR 中已经写出的最小证明

`test/test_panel_vertex_geometry_vjp.py` 使用现有 `panel/kernel.py`，固定八面体连通性；全矩阵源/双层势、`jnp.linalg.solve` 和外部采样点势都真实依赖六个三维顶点。其任意外部势的上游种子经过 `jax.vjp` 返回 `[6,3]` 顶点梯度，再以三档中心差分验证随机/定向变形。

这是**非升力/势场目标**的几何 VJP，不能当作厚翼身局部 Cp/力/矩的 VJP 验收。独立 Python 环境 float64、JAX 0.9.0.1 的最小证明已手动运行：方向 AD `1.40746085805e-3`，差分 h=1e-4 `1.40746085663e-3`，h=1e-5 `1.40746085741e-3`。这一小测试不是用户本地 JAX 0.10.2 环境的回执；测试必须在仓库验证后才能视为通过。

运行：

```bash
python -m pytest -q test/test_panel_vertex_geometry_vjp.py
```

## 5. 下一阶段任务的推荐顺序

1. **P0**：本 PR 新测试通过，并在现有 `panel/solve.py` 外抽取可导入纯 JAX 求解函数；固定旧 CLI 契约与保存字段不变，比较原来的三参数路径数值一致性；
2. **P1**：对第一个固定翼身 fixture，新增一个明确、不会改变 TE/contact 拓扑的机身 xyz 或机翼 chord/placement 节点变形，沿**局部压力力/力矩**输出用任意 VJP 种子做三档 FD；
3. **P2**：包括 y/展长的几何变形，证明 wake_basis、te_basis、cut/contact 的参数化在变形后正确；不满足则限制当前接口到 xz-only family，而不是算错梯度；
4. **P3**：对接 GRASS 生成器的 Torch CST/机身参数化：先采用 CPU `torch.autograd.Function` → JAX `vjp` → Torch 反向的显式形状/精度契约，再用随机参数方向验证整个链；
5. **P4**：等压力积分与环量/独立场审计达到设计精度后才允许用作主要气动物理损失。大规模 batch 内存和加速随后单独调度。

## 6. AeroSandbox 的重要借鉴与边界

AeroSandbox 的 WingXSec 等几何变量通过 `aerosandbox.numpy` 生成薄翼面坐标，`VortexLatticeMethod.run()` 随后以 CasADi traceable 数学组装并用 `np.linalg.solve` 求环量，因此可实现 `机翼设计变量 → 机翼薄面 VLM 总气动力` 的导数。源：https://github.com/peterdsharpe/AeroSandbox/blob/master/aerosandbox/aerodynamics/aero_3D/vortex_lattice_method.py

它的 `VortexLatticeMethod.run()` 只把 `airplane.wings` 加入影响矩阵，**不会在 VLM 的 AIC 中联合加入有厚度的机身外表面**。另一类 `AeroBuildup` 的机身气动基于经验模型，不等同于翼身三维联合面元。其 `Wing.mesh_thin_surface()` 生成固定分辨率/连通关系的薄翼面，几何坐标可以是 symbolic CasADi；显式布尔/重网格同样不能自动用静态图求导。

作者在 https://github.com/peterdsharpe/AeroSandbox/discussions/110 中明确解释，直接把原始 Airfoil.coordinates 作为优化变量可能改变 LE 索引，使静态图失效，推荐使用 KulfanAirfoil 等保持参数化拓扑的方法。对 Dpanel/GRASS 的启发就是先固定拓扑，在合法参数域内对**坐标映射**求导，而不把离散索引判断视为可微函数。
