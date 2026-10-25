# OCC 正半域半模 CAD 网格：从 STEP 直接到 half-native

**状态：已提交独立 CAD 网格实现，尚待 Windows/Gmsh 真实 STEP 算例验收。** 依赖 Dpanel PR #21 中已经本地通过 8 项测试的右半模积分核；本分支不修改旧全模 `panel/cad.py`、`panel/build_surface.py`、`panel/solve.py` 或 GRASS 默认气动。

## 路线

`panel/cad_half.py` 在 OCC 内复用完整对称的 wing.step（以及可选 body.step）、原始 TE JSON、`lifting3d/cad_components.py` 源曲面分类和 `cad_contact.py` 的流体侧接触发现。它**先创建 y>=0 的半个外流场**，从该流场扣除翼身真实并集，再与非负 y 的有限 z=0 尾迹碎分，使用 Gmsh **只对真正匹配源 CAD 的正半侧外露壁面和流体侧尾迹三角化一次**。OCC 产生的平面 y=0 临时流体封盖必须存在，但不进入 wall 标签。无全文网格先剖分后截半，无左右独立节点配对/重心焊接。

`panel/half_surface_mesh.py` 以同一组 Gmsh 节点 TAG 建立壁面、尾迹、外露 TE 和 contact cut 的连接关系。区别于原先必须闭合的全壁面：正半侧壁面**仅允许**在 y=0 存在开放边环；禁止跨平面三角、三个点全在 y=0 的假 cap、非流体侧 wake、不共形节点及奇异有向体积。壁面按边传播一致绕序，再依据镜像后正物理体积选择全局外法向。镜像用于拓扑/体积审计，不创建另一套气动未知量。

严格使用真实 wake 边与两侧 wall bank 共有的面片边识别 TE；wake 在身侧接触造成的非 TE 边也记录为 `cut_edges`，在后处理构造压力邻域时同样不跨此边拟合。镜像保持 upper→upper、lower→lower，不能沿用旧误判 upper→lower。

半模 `gamma_stations` **从 y=0 开始**，末端机翼 tip gamma=0，但半模根部 gamma **不是**零：无身侧接触时根部是独立自由度；若 y=0 到某接触端属于同一壁/尾迹 contact 片，`half_contact_station_map` 使对应 DOF 合并。该根部约束不同于全翼两端 `-1` 的旧映射。

结果为项目内 `runs/.../surface.json`，schema `dpanel.half_planar_cut.v1`，可以直接调用 `python -m panel.run_half_native`（PR #21）。其中 `reference_area` 必须是**完整飞机**平面参考面积；`reference_chord` 和原点来自源 TE 声明，`freestream` 与 `density` 由 CAD CLI 显式运行参数决定；输出力/矩使用 SI。半模内核瞬时积分镜像三角作用，不保留独立左侧 `mu`、`gamma` 或全机 AIC。

## 新文件与扩展

- `panel/cad_half.py`：OCC/STEP 正半域 fluid-cut、来源物面分类、wake fragment、contact 端点加站位、Gmsh 半域一次剖分、只读失败日志，产物 `surface.json` / `manifest.json` / `gmsh.log`。
- `panel/half_surface_mesh.py`：纯 NumPy 的半网格拓扑导出、对称面开放边环/有向体积、真实 TE banks、wake 延长、根部 contact DOF 映射。
- `panel/half_native.py`：扩展已验证压力邻接器，允许显式 `cut_edges`：尾缘与机身 wake-contact 两种不连续边都不能跨过去拟合局部切向势梯度；保持原 `stencil_rings` 默认逻辑。
- `panel/run_half_native.py`：新 schema 需要明确的 `cut_edges` 字段；之前的小厚翼示例已更新提供与 `trailing_edges` 相同的两条切边。
- `test/test_half_surface_mesh.py`：纯网格导出/半侧内核读入、半模根部 contact、伪 cap、跨平面、断开的 Gmsh 节点及尾迹延长拒绝。

## 本地真实 STEP 冒烟与三阶段验收

先在单独 Dpanel worktree，用此前已验收的 Gmsh runtime（`audit/runtime/python`）和可用的 JAX 0.10.2 / SciPy / NumPy 执行；无需新依赖或改训练环境。第一项单翼用仓库已有的真实 full wing.step，**但只生成右半域网格**：

```powershell
python -m pytest -q test/test_half_surface_mesh.py test/test_half_native_panel.py

python -m panel.cad_half `
  --wing-cad lifting3d/results/cad_inputs_v1/naca4412_wing_body/wing.step `
  --trailing-edge lifting3d/results/cad_inputs_v1/naca4412_wing_body/trailing_edge.json `
  --size 0.3 --far-size 1.5 --radius 5 --wake-strips 8 `
  --out runs/half_cad_wing_only_v1

python -m panel.run_half_native `
  --surface runs/half_cad_wing_only_v1/surface.json `
  --out runs/half_cad_wing_lu_v1 --mode lu
```

第二项添加机身，仍只剖分一个正半域**翼身 union 的真正外露蒙皮**：

```powershell
python -m panel.cad_half `
  --wing-cad lifting3d/results/cad_inputs_v1/naca4412_wing_body/wing.step `
  --body-cad lifting3d/results/cad_inputs_v1/naca4412_wing_body/body.step `
  --trailing-edge lifting3d/results/cad_inputs_v1/naca4412_wing_body/trailing_edge.json `
  --size 0.3 --far-size 1.5 --radius 5 --wake-strips 8 `
  --out runs/half_cad_wingbody_v1
```

第三项：从同一新半模 fixture 用 `panel.run_half_native` 分别跑 LU 与 GMRES，小规模开始，之后再运行结构、接触/尾迹与独立势场审计。完整全机网格只能在验证对称核的**测试**中临时镜像；旧的左右各自 Gmsh 剖分不是相同离散，不能要求旧/新 Cp 逐面为零差。

每个步骤都保留`failure.json`（含失败 stage 和 traceback），不覆盖旧 `runs`；请报告 **真实模块路径、精确 SHA、gmsh 版本、wall/wake/TE 数量、symmetry seam 节点/开放边、wall components/正体积、contact intervals、gamma 自由度、是否出现 fake cap/crossing**，以及半模 LU 的真残差、Cp/面力/矩、CPU 耗时和峰值 RSS。后续回归需对完全一致的新半模拓扑对拍临时镜像全模，比较 `A_{half}=RAE`、原始 Cp、分量载荷与几何/站位 VJP/FD。

## 主要未验证项

当前云端执行环境没有 Gmsh，**没有真正运行 STEP→半域 OCC→Gmsh 的端到端测试**；新代码必须取得本地真实报告后才能宣称半域 CAD 通过。尤其要关注半域对称截面边界的 OCC 面分类、因机身而产生的 `smooth_wall_contact`、与真实外露 TE 的对应以及严格共形的 Gmsh 节点，不能为了让其通过而按几何距离把不同节点焊起来。

另外，本模块**不对 OCC Boolean/重新网格本身求导**；只有生成后的合法固定拓扑半模点坐标可通过 PR #21 的 JAX 求解器求导。半模 root/contact 合法域、完整翼身共形几何和压力精度仍需单独证据。GMRES 当前仍可能比 LU 慢，`physical_accuracy_accepted=false` 始终保留，不启动正式 GRASS 整机训练。
