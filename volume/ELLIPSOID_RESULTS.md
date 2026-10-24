# 原生 CAD 椭球解析核验

## 预注册

目标几何为 Gmsh OpenCASCADE 球体经 `dilate` 得到的闭合 prolate 椭球，原生半轴 `(3,1,1)`、中心 `(2,-3,1)`。预定 builder 参数为 `size=.06`、`curvature=32`、`R=5`、`far-size=1`、`NCPU=8`。同一固定网格运行 `M=0, alpha=0/5`，`Sref=1`，均匀外边界条件，不重建几何或网格。直接导出的单体 STEP 经 builder 后未产生 tet10；为保留验证机会，实跑改用同一 sphere+dilate 实体在 `x=2` 处分成两个闭体的 STEP，由 builder fuse。这个变化、builder bbox 偏移及 OCC 警告均保留为未通过项，不把 workaround 宣称为原定流程完全通过。

独立参考只在分析阶段使用。椭球退极化因子按

`N_i = abc/2 * integral_0_inf ds / ((a_i^2+s) * sqrt(product_j(a_j^2+s)))`

计算，要求 `abs(sum(N)-1) < 1e-10`。单位球作为解析 patch：`N_i=1/3`，`q=1.5*(U-(U dot n)n)`，并核验椭球切向速度公式退化到球体公式，误差小于 `1e-12`。

对实际 P2 物面三角形的三阶积分点直接采样。将原生半轴除以 builder 记录的 `native_length`，中心用 `(native_center-native_origin)/native_length` 变换；由采样点的椭球法向 `n ∝ (x-center)/axes^2` 计算 `q=(I-n n^T)@(U/(1-N))` 和 `Cp=1-|q|^2`。计算面积加权 Cp RMS、raw wall 法向速度 RMS、数值面法向与参考法向差异，以及点处椭球 level-set 偏差 `sum(((x-center)/axes)^2)-1`。解析速度不施加到边界条件或数值解。

预先物理门禁（alpha=0 与 5 各自均须满足）：

- 面积加权 Cp RMS `< .03`。
- raw wall normal velocity RMS `< .03`。
- 压力积分 `abs(CL) < .005` 且 `abs(CD) < .005`。

另核对原生 bbox 归一化映射是否满足预期原点 `(2,-3,1)` 和最大边长 `6`，误差 `<1e-5`。这是几何/坐标映射要求，不与物理误差门禁混并；不通过时仍报告物理指标，但整项工作流标为未完全通过。上述条件只针对指定 CAD、网格、外边界和 `M=0`，不构成任意 CAD 或有限外域误差证明。椭球解析参考不得用于 `M>0`。

## 解析与离散定义

来流单位向量为 `U=(cos(alpha),0,sin(alpha))`。对齐椭球的表面法向由实际曲 P2 物面采样位置计算。该解是不可压、无环量、无限域椭球外流解析解；实际计算域为 `R=5` 的立方外边界并施加均匀势 Dirichlet，因此有限域截断和 P2 离散误差均保留在比较中。

另对历史 compact 椭球的 `M=0` 快照执行同一分析作为旁证；不得将其 `M>0` 结果解释为该不可压解析参考。

## 实测

2026-10-23。驱动为 `python volume/test_cad_ellipsoid.py`，实际运行由 [`test_cad_ellipsoid.py`](/D:/3D/Projects/panel/volume/test_cad_ellipsoid.py) 记录。逐例 JSON、日志、STEP、固定网格及完整 solver 输出保存在 [`results/cad_ellipsoid_v1`](/D:/3D/Projects/panel/volume/results/cad_ellipsoid_v1/)。

| 工况 | solve 状态/退出码 | Cp RMS | raw wall `vn` RMS | 压力 CL | 压力 CD | level-set max abs |
|---|---:|---:|---:|---:|---:|---:|
| M=0, alpha=0 | converged / 0 | 0.00237858 | 0.00403296 | -3.77e-7 | 2.07e-6 | 7.19e-5 |
| M=0, alpha=5 | converged / 0 | 0.00254994 | 0.00435691 | 4.43e-7 | 1.97e-6 | 7.19e-5 |

两例均满足预注册物理门禁：Cp RMS `<.03`、raw wall `vn` RMS `<.03`、`abs(CL/CD)<.005`。两例共用 mesh SHA256 `78951b17d237bdb8573df6a0bde849bb8c24f4c9897e3af401bfbbae97665cc2`；每例线性残差约 `4.4e-13`，求解分别约 22.28/22.71 s。退极化因子为 `(0.1087094651, 0.4456452675, 0.4456452675)`，求和误差 `2.22e-16`。单位球 patch 得 `N_i=1/3`，球面切向速度误差为 0，Cp 误差 `3.33e-16`。

**仍失败/偏离项**

- 直接的单体 sphere+dilate STEP 保留于 `results/cad_ellipsoid_v1/ellipsoid/`。builder 退出码 2，报告 `No quadratic tetrahedra were generated`；日志显示 body surface 无元素、3D region 判为 void。
- 实际可网格化的分体 STEP 保留于 `split_step_replay_v2/ellipsoid/input.step`。builder 将两个导入闭体 fuse 后得到 39,625 points、25,950 tet10，但 OCC 切分日志有 `BOPAlgo_AlertUnableToOrientTheShape` 警告。
- builder 的 STEP B-spline bbox 不是几何椭球 bbox：记录 `native_origin=(0.5,-3,1)`、`native_length=9.0000002`，而目标应为 `(2,-3,1)`、`6`。因此归一化半轴按 builder 长度是 `(0.33333333,0.11111111,0.11111111)`，实际中心变换为 `(0.16666666,0,0)`，没有实现要求的归一化中心 `(0,0,0)`。分析按实际变换中心比较，level-set max abs 为 `7.19e-5`；不得将此次结果说成满足了中心归零的坐标映射。
- 历史 compact ellipsoid 的 M=0 快照也已分析：Cp RMS `0.0250461` 通过，但 raw wall `vn` RMS `0.0371257` 超过 `.03`，因此记录为 gate failure；`abs(CL)=0.0009587`、`abs(CD)=0.0017412` 仍通过。它仅作旧网格旁证。

故结果状态为 **物理门禁通过，但原定单体 STEP 构网和 bbox 归一化中心要求未通过**。没有调宽任何门槛。解析速度只用于离线 Cp 计算；求解始终采用 uniform 外边界，raw wall 速度未作投影或修正。

## v2：单体多 patch STEP

v1 保持不变。新批次输入位于 [`cad_ellipsoid_v2`](/D:/3D/Projects/panel/volume/results/cad_ellipsoid_v2/)。没有使用两个闭体 workaround：从 v1 split halves STEP 读取三张 BSpline 曲面，排除两张中间 Plane，以三张曲面建立一个 surface loop 和一个 volume，再导出单体多 patch STEP。构造明细和 STEP hash 在 `construction.json`。这是具有显式多面片拓扑的独立输入，不代表原始单面、无 `EDGE_CURVE` 的 STEP 已被证明有效。

用当前 `build_cad_mesh.py` 对该单体 STEP 构网，参数为 `size=.06`、`curvature=32`、`R=5`、`far-size=1`、`NCPU=8`。builder SHA256 为 `e2a4ea0ae2e3892792efc0fda0c9d199366ee04f52b2816e5fdb81bbc7024221`。成功导入 1 个 solid，三张 body patch 的表面三角形数分别为 1702、770、774，生成 39,421 点、25,791 个 tet10。mesh SHA256 为 `2fd91857efe35344af27042f8d7cd2eee12c486c689b742d8ed2187dd1ef1e84`。数据和 metadata 保存在 `results/cad_ellipsoid_v2/mesh_current/`；旧 pilot `mesh/` 未覆盖。

metadata 记录 OCC conservative bbox 为 `[-4.0000001,-4.732050907569,-1e-7]` 至 `[5.0000001,-1.267949092431,2.0000001]`，`native_origin=(0.5,-3,1)`、`native_length=9.0000002`，导入体积 `12.572085134994865`。builder 明确注明该 bbox 是保守范围，B-spline control bounds 可超过几何范围；origin 是该 bbox 的中点，不保证等于几何或质量中心。故目标几何中心 `(2,-3,1)` 在实际 solve 坐标映射后为 `(0.16666666296,0,约0)`，并非零。旧协议中原生 bbox 映射原点/边长 `<1e-5` 的门槛保持 **failed**；此偏差作为 builder 的坐标映射性质记录，不被重新解释成物理求解失败，也不被物理 Cp 门槛掩盖。

在固定 `mesh_current/mesh.npz` 上串行执行 `M=0`、`beta=0`、uniform 外边界、`Sref=1`、`NCPU=8`、quadrature order 3 的 alpha 0 和 5 求解。两例 solve 退出码均为 0、状态均为 converged；mesh hash 和 solver/p2 源 hash 完全相同。每例的完整 `result.json`、`field.npz`、标准输出日志、退出码及离线 analyzer JSON 分别保存在 `m0_alpha_0_current/` 和 `m0_alpha_5_current/`。Analyzer 仅比较解析参考，没有把解析速度施加给 solver。

| 工况 | solve 残差 | Cp RMS | raw wall `vn` RMS | 压力 CL | 压力 CD | level-set max abs | 物理门槛 |
|---|---:|---:|---:|---:|---:|---:|---|
| M=0, alpha=0 | `4.53e-13` | 0.00238233 | 0.00405799 | `1.01e-6` | `4.00e-6` | `7.19e-5` | 4 项均通过 |
| M=0, alpha=5 | `4.67e-13` | 0.00255542 | 0.00437804 | `1.61e-6` | `4.25e-6` | `7.19e-5` | 4 项均通过 |

两例均满足原先不变的物理门槛：Cp RMS `<.03`、raw wall `vn` RMS `<.03`、`abs(CL)<.005`、`abs(CD)<.005`。退极化因子为 `(0.1087094651,0.4456452675,0.4456452675)`，求和误差 `2.22e-16`；单位球解析 patch 的切向速度误差为 0、Cp 误差为 `3.33e-16`。实际 P2 物面采样的椭球 level-set 面积加权 RMS 为 `1.03e-5`，最大绝对值为 `7.19e-5`。这些物理量通过不改变 bbox 映射门槛的结论。

**v2 结论与剩余限制**

- 单体多 patch STEP 在当前 builder 下有三张非空 body surface 并成功生成可求解 tet10 网格；原始单面 STEP 的 void 输入缺陷仍未被此结果追认或修复。
- 指定 M=0 椭球物理参考的四个预注册门槛在 alpha 0/5 均通过。
- 保守 OCC bbox 映射不将目标原生几何中心移到 solve 原点；坐标映射 `<1e-5` 门槛仍失败，实际映射参数和非零归一化中心如上。没有调宽门槛，也没有把 bbox midpoint 称为真实几何中心。
- 结果只覆盖该单体多 patch STEP、指定网格和 `R=5` uniform 外边界下的无环量不可压解；不认证 `M>0`、升力/尾迹、其他 STEP 拓扑、CAD 坐标映射精度或有限外域误差。
