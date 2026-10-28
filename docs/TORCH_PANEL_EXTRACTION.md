# 从 GRASS-plane 提取可微 PyTorch 面元研究内核

状态：**独立研究代码与移植测试待本地验收；不是已交付升力/整机面元求解器。**
来源：GRASS-plane master `aa6d81faf03c556db475d9edec6c025f90eb5fac`；本提取建立在 Dpanel master 上，不修改 PR #2/#5 或其本地执行环境。

## 仓库职责

- GRASS-plane：任务条件到整机几何、结构、质量/重心、配平、训练，以及消费气动结果的 adapter。
- Dpanel：势流物理算子、数值求解、尾迹/Kutta、近场/远场载荷、梯度、半模和性能诊断。
- **阶段一训练主候选继续是已有 VortexAD**；本模块保留为实验/数值基准。阶段二按真实性能瓶颈发展 Matrix-free、快速算子/伴随或学习预条件，而不是重造未经验证的同类内核。

## 已搬运的 GRASS 代码

| 新位置 | GRASS 原始文件 | 复用价值 | 当前适用边界 |
| --- | --- | --- | --- |
| `torch_panel/kernels.py` | `panel3d_kernels.py` | 解析常源/常偶极子三角面元的势影响 `[P,N]`、torch 梯度 | 非奇异点及自影响 `-1/2`，需继续核查对角 autograd、近边缘稳定与内存 |
| `torch_panel/symmetry.py` | `panel3d_symmetry.py` | XZ 镜像三角绕序与真实+镜像影响系数，减少**非升力**半模未知量 | 尚无尾迹/Kutta 镜像测试，不能标识为完整有升力半模 |
| `torch_panel/nonlifting.py` | `panel3d_nonlifting.py` | 闭合非升力体内部 Dirichlet、网格闭合/外法向与势函数评价 | 稠密 torch.linalg.solve，仅闭体非升力验证原语 |
| `test/test_torch_panel_*.py` | 4 组 GRASS 测试 | 数值积分、gradcheck、闭合性、镜像等价性 | 迁移后仍须本地重跑 |

原始 GRASS-plane 根目录持有 Apache-2.0 license；本目录已附 `LICENSE-APACHE-2.0.txt`，且提取文件保留源路径、来源 commit 与修改说明。当前 Dpanel 其他模块并不因此自动采用这一许可。初次提取只修改 imports，不修改物理公式，便于对照。

## Dpanel 自己已经存在的互补实现

- `core/triangle_kernel.py`：NumPy/Numba 解析源势/偶极子势与**场点速度梯度**；对源、双层的自面外侧迹、近面/近边缘对数奇异性做过独立试验；
- `core/test_kernel.py`、`core/test_kernel_near_field.py`：独立高阶面积积分与数值微分、球面 Gauss 恒等式、近边缘与缩放/坐标等变测试；
- `core/ACCEPTANCE.md`、`core/RESULTS.md`：球/椭球收敛和局部边界残差等报告；成功只属于对应 NumPy 非升力闭体核，不可转移为 PyTorch 的物理验收。

`test/test_torch_panel_vs_numba.py` 针对**远离奇异表面的三角形**比较两套势影响与场点梯度；Numba 参考来自 Dpanel，独立于 GRASS。待测后再逐步加入 near-edge、自面迹、场点梯度和更严格收敛比较。

## 不原样搬运的代码及原因

`GRASS-plane/panel3d_geometry.py` 依赖该仓库的 `cst_airfoil_codec.cst_surfaces_at_x` 和 `wing_geometry_encoder.wing_section_points`。GRASS Issue #19 的本地实验确认：
- `thick_wing_topology` 中 `kutta_pairs` 的下表面索引不接触尾缘边（错一条三角索引）；已验收的是几何闭合与基本梯度，**不是 Kutta/升力**；
- `explicit_full_mirror` 根据 `abs(y)<=1e-12` 猜测接缝，而非明确的根部顶点 ID；
- `thick_wing_half_mesh` 强制根部处于 y=0，无法表达翼梢小翼或非中线根部的任意挂接；
- CST 参数、截面 frame 属于整机生成器几何语义，不应复制到气动核心。

建议后续由 GRASS 提供固定拓扑、已经验收的 `vertices/faces/te_edges/component_ids/seam_ids` 和连续节点张量，Dpanel只处理通用数值几何。若最终希望 Dpanel 自己产生网格，也需要**独立的通用网格构造层**；不要把 GRASS 的训练编码器整体复制过来。

## 建议的未来对接契约（尚未实现）

```text
GRASS generator(theta, topology_family)
  -> MeshInput(
       vertices: Tensor[V,3], faces: int[F,3],
       component_ids: int[F],
       trailing_edges / adjacent_upper_lower_panel_ids,
       symmetry_plane / seam_vertex_ids,
       reference: Sref, Cref, r_ref
     )
  -> Dpanel backend (VortexAD / experimental Torch / future fast BEM)
  -> Forces, moments, panel_Cp_raw, panel_center/normal/area,
     componentwise/spanwise loads, solver_diagnostics
  -> Grass structural / trim / mission losses
```

拓扑、边缘邻接和部件标签只在合法构型族内固定，`vertices` 及连续工况参数必须能提供 VJP。对机身与多翼面，必须构造**真实暴露的公共外表面**，移除埋入实体的面元；不可将相交封闭翼面与封闭机身简单叠加求解。不得自动把几何表面积作为 Sref。

## 本地验收命令（运行在已有 PyTorch + SciPy + pytest + NumPy + Numba 环境）

```bash
python -m pytest -q test/test_torch_panel_kernels.py test/test_torch_panel_symmetry.py test/test_torch_panel_nonlifting.py test/test_torch_panel_validation.py test/test_torch_panel_vs_numba.py
```

记录 SHA、Torch CUDA/CPU、dtype、命令/退出码、首个 traceback、全部误差。不要修改已有 `aero.py`、Dpanel P2 模块或正在执行的 PR #5 基准。异常报告先诊断数学符号、自面跳跃和边/角落数值问题，不要通过隐含涡核、裁剪压力或扩大容差掩盖。

## 后续步骤

1. **源代码分离验收**：必须证明 Dpanel 的 Torch 包能在完全不安装 GRASS 的环境导入；原测试与 NumPy/Numba 交叉测试通过，许可证与源追踪完整。
2. **核心数值改进**：数值上正确的三角核之后，测试分块与定制伴随，避免反向图 `O(N²)` 中间内存过大；项目已有 GRASS RTX 5060 面元核显存结果可作性能参考，但非完整整机性能。
3. **独立升力体阶段**：先重新设计尾缘配对 / Kutta / 尾迹 / 表面速度和 Cp；沿用 Dpanel 原有解析核、VortexAD 以及 VSPAERO 的独立基准，不宣称未经认证的升力预测。
4. **GRASS 逐步依赖切换**：在后续 GRASS PR 中增加薄 adapter，先 dual-run/对照后再删除旧 panel3d 文件；不能在此 PR 直接删除 GRASS 的物理文件或改默认生成器训练路径。
