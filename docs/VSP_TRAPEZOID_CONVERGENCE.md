# 梯形翼 DPanel / VSPAERO 网格收敛对比（cases/vsp）

来源任务：用户 2026-10-27 要求「用 `cases/vsp` 的梯形翼算例，给 DPanel 和 VSP 各跑气动系数网格收敛性，画出尾迹和积分的结果；现有网格密度作为最高档，DPanel 做不到就取资源允许的最大值；参考面积和展长按实际几何重算」。

状态：`physical_accuracy_accepted = false`。本文只声明**各自网格阶梯内的数值稳定性**、**两者差异的量级与归因**，不声明任何一方的绝对值是真值。

## 1. 结论摘要

1. **几何与参考量按实际几何重算**（`cases/vsp/trap.vspgeom` 实测，而不是 `trap.vsp3` 里的陈旧 parm）：半展 1.0 m、全展 B = 2.0 m、根弦 0.2、尖弦 0.1（taper 0.5）、前缘全程 x = 0（无后掠）、后缘后掠 −5.71°、**NACA 2412**、**Sref = 0.30 m²**（全机）、MAC = 0.155556 m、AR = 13.33。旧 `trap.vspaero` 的 `Sref = 100 / Bref = 1` 与旧 `trap.polar` 均无效，未使用。
2. **CL(5°) 收敛**：DPanel 0.630726 → 0.656163 → 0.684482 → 0.694798 → **0.695979**（1004/1822/3382/6562/12290 壁面），相邻档 $-4.1\%\ /\ -4.2\%\ /\ -1.5\%\ /\ -0.17\%$，**从 3382 壁面起已在最细档的 5% 内、从 6562 起 0.2% 内**；VSPAERO 0.641479 / 0.643757 / 0.642537 / **0.640883**（240/854/3120/12480 薄片面元），**每一档都在最细档 0.5% 内**。
3. **最细档差 8.6%**：`CL_DPanel / CL_VLM = 1.0860`（+0.05510）。这是本文最稳的一条差异：两码各自的 CL 都已收敛，差值不是网格噪声。
4. **阻力两侧都不能当参考值**：DPanel 的 Trefftz CD 数值上很稳（0.009391 → 0.010276，相邻档 +0.6%~+4%），反推展向效率 $e = CL^2/(\pi AR)/C_D$ = 1.01/1.07/1.11/1.13/1.13，**超过平面翼的 Munk 极限 1**；VSPAERO 的 CDi 单调爬升 0.010381 → 0.011504 → 0.013955 → **0.015953（l3→l4 仍 +14.3%）**，反推 e 从 0.946 掉到 **0.615**，低于任何合理的薄翼展向效率。**两侧的阻力都在 10~30% 量级不可信**，只能比趋势。
5. **VSPAERO 的诱导阻力不是尾迹欠松弛**：把 `WakeIters` 5 → 20、`NumWakeNodes` 32 → 64、`FarDist` 5 → 20 重跑 l3（`runs/vsp_conv/vlm_l3w`），CDi 0.013955 → 0.013700（**−1.8%**）、CL 0.642537 → 0.641059，即尾迹设置不是 CDi 随加密上升的原因。
6. **VSPAERO 的 CDo 是无粘也不为 0 的加性项**：$\alpha = 0$ 时 CDo = 0.02046（l1）/ 0.02049（l4），$\alpha = 5°$ 时 0.02278/0.02279，几乎与网格无关；`CDtot = CDo + CDi`，且 `CDwtot ≈ CDo`。所以 **`CDtot` 与 `L/D` 不能用**，本文只比 CDi。
7. **升力线斜率**：DPanel 0.0995 /deg（5.70 /rad）、VSPAERO 0.0896 /deg（5.13 /rad）；Prandtl 有限翼 $2\pi AR/(AR+2)$ = 0.09537 /deg、Helmbold 0.09445 /deg——两者分列理论值两侧（+4.2% / −6.1%），**都不能单独当标定**。
8. **展向分辨率是 DPanel 的结构性瓶颈**：Γ 的自由度只有 16 个（站位 = 后缘结点 ∪ 尾迹条带的并集，17 个物理站位，间距最粗 0.1 半展），VSPAERO 最细档 240 条带。中展 cl 0.7221（DPanel）vs 0.6699（VLM）差 7.8%；尖部 cl 恒 0（Kutta 闭合）vs 0.2070（且 VLM 从 0.3809 → 0.2826 → 0.2087 一路降，**尖部未收敛**）。
9. **尾迹**：DPanel 是冻结（不卷起）的平面尾迹片，Γ 沿 x 严格不衰减（三个下游站剖面完全重合，`fig19`）；VSPAERO 的尾迹会卷起，但 `WriteTecplotFile` 只导出壁面 zone，尾迹几何只在 `trap.adb` 二进制里，没有可读文本。
10. **修掉一个 Trefftz 归一化 bug**（本轮）：`runs/trefftz_drag.py` 原来按 `span*chord` 归一化，传 MAC(0.155556) 会把分母放大成 0.3111 而真 Sref = 0.30，CD_T 系统性低 3.7%。已加 `area=None` 关键字，`runs/vsp_conv/dpanel_case.py` 传真实 `reference_area`；椭圆解析锚点 `elliptic_test(n=201)` 比值 0.994925。

11. **厚面元（panel）模式对照**（§4.6、§4.7、`fig21`、`fig22`）：VSPAERO 的 panel 模式（`GeomSet=SET_FIRST_USER` + `ThinGeomSet=SET_NONE`，`Model type: PANEL or MIXED`）才是与 DPanel 对等的厚面元对照。**同一 mesher（OpenVSP 3.55，`vspapi` 生成）五档** 888/1908/3388/6292/12996 面板 → $C_L(5°) = 0.71968 / 0.72118 / 0.71876 / 0.71256 / 0.70864$（最粗档在最细档 1.6% 内，但 1908 之后单调缓降、未走平）；同 mesher 的薄 VLM 五档 420/918/1650/3094/6422 → 0.64103 / 0.64468 / 0.64404 / 0.64083 / 0.64018（最细两档差 0.10%）。三分法最细档：DPanel **0.695979**（12290 壁面）、panel **0.70864**、VLM **0.64018** → DPanel 比厚面元 VSPAERO 低 **1.8%**、比薄 VLM 高 **8.7%**，panel 比 VLM 高 **10.7%**。
12. **VSPAERO panel 模式的绝对 CL 依赖 mesher 版本（重要警告）**：同一个 `.vsp3` 模型、同一个 `vspaero.exe`、同一份 `.vspaero`，**只用 OpenVSP 3.52 与 3.55 分别剖分**（6222 vs 6344 面板，体积/面积/体积质心差 <0.1%）→ $C_L(5°) = 0.6544$ vs $0.7126$（**+8.9%**）、$C_L(0°) = 0.1695$ vs $0.2259$（+33%），且差值是一个纯截距（两者升力线斜率同为 5.6 /rad）。3.52 那族网格的弯度升力即使到 6222 面板也没收敛（$C_L(0)$ 0.0339→0.1695 一路爬），3.55 那族在 888 面板就平了（0.2304→0.2260）。**所以 panel 模式的阶梯必须由同一个 mesher 生成，且不宜跨版本引用绝对 CL。** 另外 `CLi` 列在 `LECluster = 0.25` 的网格上会给出 $10^2$–$10^4$ 倍错数（`docs/VSPAERO_PANEL_POLAR_DIAGNOSIS.md` §4.3），只能用 `CLwtot` / `trap.lod`。

## 2. 几何与参考量

| 量 | 值 | 来源 |
| --- | --- | --- |
| 半展 / 全展 | 1.0 m / 2.0 m | `trap.vspgeom` 顶点 bbox（y ∈ [0,1]，`symmetry = 1`） |
| 根弦 / 尖弦 | 0.2 m / 0.1 m | `x_max = 0.2 - 0.1 y` 在 y = 0 / 1 |
| 前缘 / 后缘后掠 | 0° / −5.71° | 所有站位 `x_min = 0`；`x_max` 线性下降 |
| 翼型 | NACA 2412（camber 0.02 @ 0.4c，厚 0.12） | `trap.vsp3` `XSecCurve`（shape 7 = XS_FOUR_SERIES） |
| Sref（全机） | 0.30 m² | $2 \int_0^1 c(y) dy$ = 2 × 0.15 |
| MAC | 0.155556 m | 线性弦长分布 $\frac{2}{3} c_r (1+t+t^2)/(1+t)$，t = 0.5 |
| AR | 13.33 | $B^2 / S_{ref}$ |
| Vinf / rho / Re_MAC | 20 m/s / 1.225 kg/m³ / 1.99e5 | 两码共用 |

DPanel 的 `reference_area` **必须是全机面积**（`panel/half_native.py:241 _half_loads` 的 `all_f = one_f + one_f*_REFLECT` 已把右半力×2），写半模面积会让 CL 翻倍——本轮踩过一次。

## 3. 数值设置

### 3.1 DPanel（半模厚面元，`panel.cad_half` + `solve_half_fields`，JAX x64 CPU）

| size（声明弦长单位） | 物理边长 ≈ | 壁面 | 尾迹 | 未知量 | 5 迎角总耗时 | 峰值内存 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.20 | 0.0311 m | 1004 | 560 | 1020 | 2.9 s | 未测 |
| 0.14 | 0.0218 m | 1822 | 642 | 1838 | 6.7 s | 未测 |
| 0.10 | 0.0156 m | 3382 | 938 | 3398 | 20.2 s | 未测 |
| 0.07 | 0.0109 m | 6562 | 1455 | 6578 | 80.8 s | 未测 |
| 0.05 | 0.0078 m | 12290 | 2508 | 12306 | 262.3 s | 未测 |

几何由 `runs/vsp_conv/make_trap_cad.py` 生成的全展 STEP（21 段直纹放样、共用 NACA-4 造型、尖 LE/TE）。`panel/cad_half.py:93` 要求后缘声明半展 < `radius - size`（后缘按 MAC 归一化后半展 6.43 弦），必须 `--radius 20 --far-size 4`，默认 5.0/1.5 直接 `ValueError: Invalid CAD meshing/flight parameters`；重跑须先删输出目录（`:363` 拒绝覆盖）。迎角只换 `freestream`，几何/求解器实例复用。

### 3.2 VSPAERO（VLM 薄片，DEP-OPT 验证过的代码路径 `runs/vsp_conv/dep_vsp.py`）

| 档 | chordwise × spanwise | 薄片面元 | 名义边长（弦向 / 展向） | 展向条带 |
| --- | --- | --- | --- | --- |
| l1 | 13 × 30 | 240 | 0.0154 / 0.0333 m | 30 |
| l2 | 26 × 61 | 854 | 0.0077 / 0.0164 m | 61 |
| l3 | 52 × 120 | 3120 | 0.0038 / 0.0083 m | 120 |
| l4 | 104 × 240 | 12480 | 0.0019 / 0.0042 m | 240 |

`Sref 0.30 / Cref 0.15 / Bref 2.0 / X_cg = 0 / Symmetry = 1 / WakeIters 5 / NumWakeNodes 32 / FarDist 5`，迎角 0…10° 五档。**`cases/vsp/trap.vspgeom`（6249 面）是双面厚网格**，直接喂 VLM 会得到 CL ≈ 27 的无意义解；薄片 VLM 的等价网格是 3120 面（l3 = 用户的目标密度）。

### 3.3 目标密度与资源上限

`trap.vspgeom` 的弦向边长 0.0038 m ≈ DPanel `size 0.024`。按 $N \propto size^{-1.46}$ 外推约 3 万壁面，稠密 LU 峰值内存按 $O(N^2)$ 外推需要约 47 GB（本机可分配上限 14.45 GB），**所以 DPanel 侧最高档只做到 12290 壁面**（物理边长是目标密度的 2.0 倍，面元数是 3.9 倍）。这是资源上限，不是精度上限。

## 4. 结果

### 4.1 收敛表（`runs/vsp_conv/figs/fig16_vsp_convergence.png`）

| 档 | 面板 | CL(5°) | CL 相对最细 | CD(5°) | CD 相对最细 |
| --- | --- | --- | --- | --- | --- |
| DPanel 0.20 | 1004 | 0.630726 | −9.38% | 0.009391（Trefftz） | −8.61% |
| DPanel 0.14 | 1822 | 0.656163 | −5.72% | 0.009637 | −6.22% |
| DPanel 0.10 | 3382 | 0.684482 | −1.65% | 0.010055 | −2.15% |
| DPanel 0.07 | 6562 | 0.694798 | −0.17% | 0.010218 | −0.56% |
| DPanel 0.05 | 12290 | 0.695979 | 0 | 0.010276 | 0 |
| VSPAERO l1 | 240 | 0.641479 | +0.09% | 0.010381（CDi） | −34.93% |
| VSPAERO l2 | 854 | 0.643757 | +0.45% | 0.011504 | −27.88% |
| VSPAERO l3 | 3120 | 0.642537 | +0.26% | 0.013955 | −12.52% |
| VSPAERO l4 | 12480 | 0.640883 | 0 | 0.015953 | 0 |

读法：**CL 收敛快、阻力收敛慢，而且两条阻力阶梯的方向相反**——DPanel 的 Trefftz CD 单调升但每档 +0.6%~+4%，VSPAERO 的 CDi 单调升且 l3→l4 仍 +14.3%，意味着要声明 CDi 收敛得再往上加至少一两档（$N \gtrsim 4\times 10^4$）。按用户「系数至少收敛到 5%」的口径：**DPanel 的 CL 从 3382 壁面起合格、CD 从 3382 起合格；VSPAERO 的 CL 每档都合格、CDi 到 12480 面仍不合格**。

### 4.2 升力线与力矩（`fig18_vsp_alpha.png`）

| 档 | CL(0) | CL(5°) | dCL/dα 中心差分 @5° | 线性拟合斜率 | 截距 |
| --- | --- | --- | --- | --- | --- |
| DPanel 0.20 | 0.2000 | 0.630726 | 0.0851 /deg | 0.0847 | 0.2033 |
| DPanel 0.14 | 0.1967 | 0.656163 | 0.0910 | 0.0906 | 0.1993 |
| DPanel 0.10 | 0.1993 | 0.684482 | 0.0961 | 0.0958 | 0.2023 |
| DPanel 0.07 | 0.1955 | 0.694798 | 0.0991 | 0.0989 | 0.1978 |
| DPanel 0.05 | 0.1948 | 0.695979 | 0.0995 | 0.0994 | 0.1972 |
| VSPAERO l1 | 0.1767 | 0.641479 | 0.0925 | 0.0924 | 0.1781 |
| VSPAERO l2 | 0.1845 | 0.643757 | 0.0913 | 0.0913 | 0.1859 |
| VSPAERO l3 | 0.1882 | 0.642537 | 0.0903 | 0.0903 | 0.1896 |
| VSPAERO l4 | 0.1894 | 0.640883 | 0.0896 | 0.0896 | 0.1907 |

CL(0)（弯度升力）两码差 0.0054（2.8%）：DPanel 恒定 0.195~0.200、VLM 从 0.1767 单调升到 0.1894（+7.2%）仍未平——VLM 在粗网格上把 2412 的弯度效应抹平了。

力矩必须先统一参考点：VSPAERO `X_cg = 0`（前缘平面）、DPanel `moment_reference = [0.0388889, 0, 0]` = MAC 1/4 弦。平移 $CM_{qc} = CM_y + C_L x_{qc}/C_{ref}$（VLM 用 `Cref = 0.15`）后：DPanel −0.0633 → −0.0483（几乎不随 α 变，焦点压在 1/4 弦附近）；VSPAERO −0.1767 → −0.2122 原值，平移后 +0.011 → −0.0400，**VLM 的力矩随网格从正变负，4 档差 0.05（≈1.3 个 Cref/10）**，属于未收敛量。

### 4.3 跨度载荷（`fig17_vsp_span_loading.png`）

- 形状：中展 cl 0.7221（DPanel 最细）vs 0.6699（VLM 最细），差 7.8%；两条曲线在整个半展上都很平（比椭圆更方），符合 taper 0.5 + 无后掠。
- DPanel 的 Γ 只有 **17 个物理站位 / 16 个自由度**的 hat 基（`[0, 0.1, 0.125, 0.2, 0.25, 0.3, 0.375, 0.4, 0.5, 0.6, 0.625, 0.7, 0.75, 0.8, 0.875, 0.9, 1.0]`，由后缘结点 ∪ 尾迹条带并集得到，最粗间距 0.1 半展），**加密三角面元不增加展向自由度**：1004 → 12290 壁面，站位仍是 17 个。这是 DPanel 的 CL/阻力差异的**结构性来源**，与 issue #42 的网格研究方向一致。
- 尖部：DPanel 最后一个站位 dof = −1（Kutta 闭合把 Γ 拉到 0），cl(1.0) = 0；VLM 尖部 cl = 0.3809/0.2826/0.2087/0.2070（**从 0.38 掉到 0.21 仍未平**），所以尖部是两者差得最多的地方（相隔 0.21）。
- 中间面板「相对自己最细档的偏差」：VLM 的 240 条带除尖部外全在 ±0.5% 内（尖部到 +4.5%）；DPanel 的粗档在 −2%~−11% 之间摆动且不完全单调，说明**DPanel 的展向载荷是点态未收敛的**，只有积分量（CL、总阻力）进了 5%。

### 4.4 尾迹（`fig19_vsp_wake.png`）

DPanel 侧画的是 `dp_s005/state.npz` 的 2508 个尾迹三角形：机翼用 Cp 上色、尾迹片用 Γ 上色（`x` 限 0–0.7、`z` ±0.12）。右侧三个下游站 x = 0.21 / 0.40 / 0.70 的 Γ(y) 阶梯剖面**完全重合**：DPanel 的尾迹是**冻结**的（不卷起、不衰减），根段 Γ ≈ 1.32 m²/s → 尖部 0.37 m²/s。这是薄涡格 + 固定尾迹的必然结果，不是数值误差。

VSPAERO 的尾迹是自由尾迹（会卷起），但 `WriteTecplotFile = 1` 只写壁面 zone；尾迹只在 `trap.adb`（二进制）里，本轮没有解析，所以**尾迹几何的直接对比缺失**——这一点在 §7 边界里明写。

### 4.5 压力分布（`fig20_vsp_cp_maps.png`）

把坐标映射到 $x/c(y)$ 让平面变矩形后做 70×90 的 `griddata` 插值（DPanel 用 `(Cp_upper + Cp_lower)/2`，按 `normals_z > 0.2 / < -0.2` 分面）：

- 平面平均：DPanel −0.2261、VSPAERO 薄片 −0.3055（薄片单面 Cp 与厚面元上下平均不是同一个量，只比形状）；
- 前缘 $x/c < 0.1$：DPanel 更负（厚面元把前缘吸力峰解析出来了），中后弦两者几乎重合；
- 差值图在 $x/c \approx 0.05$ 有一条深蓝带（DPanel 低 0.4~0.5），对应 LE 处的解析差异，不是网格噪声。

### 4.6 厚面元（panel）模式：mesher 版本敏感性（OpenVSP 3.52 网格，`fig21_vsp_panel_mode.png`）

> **这一节的绝对值已被 §4.7 取代**：这里的四档网格是 OpenVSP **3.52** 剖分的，同一模型换成 3.55 剖分后 CL(5°) 从 0.6544 变 0.7126（+8.9%）。本节保留下来作为「panel 模式对 mesher 敏感」的证据。

VSPAERO 的模式由几何集成员决定（机翼进 `SET_FIRST_USER` 且 `ThinGeomSet=SET_NONE` → `Model type: PANEL or MIXED`、6249 条涡环方程；`SET_ALL` 不代表 panel，见 `docs/VSPAPI.md`），此时它和 DPanel 一样是**厚面元**，才是与 DPanel 对等的对照。但 panel 模式的 polar 有一个已知缺陷（`docs/VSPAERO_PANEL_POLAR_DIAGNOSIS.md`）：`CLi` 那一列在粗网格上会给出 $10^2$–$10^4$ 倍的错数，可用的是 `CLwtot` 与 `trap.lod` 的条带载荷。

用 DEP-OPT 的 `create_wing` 重建同一机翼的厚网格四档（52×120 为目标档、与 `cases/vsp` 的 6249 面板同量级），统一设置 `WakeIters 5 / NumWakeNodes 8 / FarDist −1 / WakeRelax 1 / Symmetry 1`：

| 档 | 面板 | CL(5°) | 相对最细 | CL(0°) | CDwtot(5°) | lod $\sum C_l \Delta y$ | `CLi`(5°) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 24×36 | 874 | 0.5242 | −19.9% | 0.0339 | 0.0173 | 0.5505 | 29810 |
| 34×52 | 1890 | 0.6193 | −5.4% | 0.1277 | 0.0204 | 0.6295 | 1607 |
| 44×76 | 3354 | 0.6439 | −1.6% | 0.1542 | 0.0213 | 0.6478 | 44.8 |
| 52×120 | 6222 | **0.6544** | 0 | 0.1695 | 0.0216 | 0.6732 | **0.635** |

读数：

- **厚面元的网格敏感性远高于薄 VLM**：panel 模式要 ~3354 面板才进自己最细档的 5%，VLM 240 面板就已经在 0.5% 内；DPanel 需要 3382 壁面进 5%（与 panel 模式巧合地接近）。
- **三分法下的 $C_L(5°)$**：DPanel 0.6960 > panel 模式 0.6544 > 薄 VLM 0.6429。DPanel 比厚面元 VSPAERO 高 6.4%、比薄 VLM 高 8.3%；厚面元 VSPAERO 比薄 VLM 高 1.8%。
- **零升力（$alpha = 0$）升力最暴露差异**：DPanel 0.1948 ≈ VLM 0.1894（差 2.8%），panel 模式 0.1695（**比 DPanel 低 13%**，833 面板时只有 0.0339，随网格持续爬升、最细档仍未平），说明 panel 模式的弯度升力要更密的网格。
- **panel 模式的阻力不可用**：`CDwtot` 0.0216 反推 $e = 0.69$（VLM 最细档同病），且 `CDwtot` 里含被污染的诱导项；只有 `CLwtot` 与 `lod` 的 $\sum C_l \Delta y$ 是自洽的（两者差 2.9%）。
- **`CLi` 的坍缩不是设置问题，是 `LECluster`**：`WakeIters`/`NumWakeNodes`/`FarDist`/`WakeRelax`/`FixedWakeFlag`/`Symmetry` 全试过，逐位不变或只挪 15%；同样的设置喂 `cases/vsp/trap.vspgeom` 得 `CLi = 34.2494`（复现 GUI），喂重建的同密度网格得 `0.6346`。二分到根因：**`LECluster = 0.25`（用户模型的值）就爆，`LECluster = 0.8` 就正常**，`TECluster` 与 `MaxGrowth` 都无关（`LE 0.25/0.50/0.80` → `CLi 125.5/5.50/0.635`，见 `docs/VSPAERO_PANEL_POLAR_DIAGNOSIS.md` §4.3）。
- **可操作结论**：要用 panel 模式做对照，请用 `VSPAEROComputeGeometry` **重新生成**厚网格（或换掉 `cases/vsp/trap.vspgeom`），并只看 `CLwtot` / `lod`；`cases/vsp/trap.polar` 里 `CLi` 列（27.3/34.2/34.2）与由它导出的 `CDtot`、`L/D`、`E` 一律不要用。

### 4.7 同一 mesher（OpenVSP 3.55，`vspapi`）的三方法阶梯（`fig22_three_way_same_mesher.png`）

§4.6 的网格族受 mesher 版本影响（见上），所以对照阶梯改用 **OpenVSP 3.55 + `vspapi`** 一次性生成全部 VSPAERO 网格，DPanel 沿用 §3.1 的五档。VSPAERO 侧统一设置：`WakeIters 5 / NumWakeNodes 8 / FarDist −1 / WakeRelax 1 / Symmetry 1 / ReCref 1e7 / Vinf 100 m/s / rho 0.002377`（panel 与 thin 同一份配置，只换几何集）。

**厚面元（panel）**：

| 档 | 面板 | CL(0°) | CL(2.5°) | CL(5°) | 相对最细 | `lod` ΣClΔy | 耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 24×36 | 888 | 0.23038 | 0.47536 | 0.71968 | +1.00% | 0.70950 | 9 s |
| 34×52 | 1908 | 0.22976 | 0.47583 | 0.72118 | +1.21% | 0.71031 | 31 s |
| 44×76 | 3388 | 0.22809 | 0.47359 | 0.71876 | +0.87% | 0.70713 | 83 s |
| 52×120 | 6292 | 0.22598 | 0.46996 | 0.71256 | +0.55% | 0.69936 | 212 s（`-omp 16` 时 44 s） |
| 74×170 | 12996 | 0.22330 | 0.46629 | **0.70864** | 0 | 0.69476 | 111 s（`-omp 16`） |

**薄 VLM**：

| 档 | 面板 | CL(0°) | CL(2.5°) | CL(5°) | 相对最细 | `lod` ΣClΔy | 耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 24×36 | 420 | 0.18124 | 0.41144 | 0.64103 | +0.03% | 0.63310 | 2 s |
| 34×52 | 918 | 0.18569 | 0.41546 | 0.64468 | +0.60% | 0.63682 | 5 s |
| 44×76 | 1650 | 0.18680 | 0.41569 | 0.64404 | +0.50% | 0.63589 | 12 s |
| 52×120 | 3094 | 0.18664 | 0.41412 | 0.64083 | +0.10% | 0.63239 | 27 s |
| 74×170 | 6422 | 0.18778 | 0.41431 | **0.64018** | 0 | 0.63155 | 18 s（`-omp 16`） |

**三方法最细档汇总**（DPanel 取 12290 壁面 / size 0.05）：

| 方法 | 面元 | CL(0°) | CL(5°) | dCL/dα（0→5°，/rad） | α₀（度） | 阻力 5° |
| --- | --- | --- | --- | --- | --- | --- |
| DPanel（厚面元，JAX） | 12290 壁面 | 0.1948 | 0.695979 | 5.743 | −1.95 | CD_T 0.010276（Trefftz） |
| VSPAERO panel（厚面元） | 12996 | 0.22330 | 0.70864 | 5.562 | −2.30 | CDwtot 0.0216 |
| VSPAERO VLM（薄片） | 6422 | 0.18778 | 0.64018 | 5.184 | −2.07 | CDi 0.0160 |

读数：

- **收敛速度：panel 888 面板即 ±1.2%，VLM 420 面板即 ±0.6%**。所以「厚面元要 3354 面板才进 5%」是 3.52 网格族特有的（§4.6），不是 panel 模式的性质。DPanel 要达到 5% 需要 3382 壁面（`fig16`），比 VSPAERO 更吃网格。
- **再加密一档（74×170）后 panel 模式并未走平**：0.72118（1908）→ 0.71876 → 0.71256 → **0.70864**（12996，6292→12996 仍 −0.55%），趋势是缓慢下降，外推收敛值约 0.705；VLM 则已平（0.64103/0.64468/0.64404/0.64083/0.64018，最细两档差 0.10%）。所以「panel 模式在目标密度附近收敛」只能说到 1% 级。
- **三分法 CL(5°)：VSPAERO panel 0.70864 > DPanel 0.695979 > VSPAERO VLM 0.64018**。DPanel 比厚面元 VSPAERO 低 **1.8%**、比薄 VLM 高 **8.7%**；panel 比 VLM 高 10.7%。
- **`-omp N` 是唯一有效的并行开关**：`vspaero trap` 1 线程 66.1 s / `-omp 8` 14.2 s / `-omp 32` 11.0 s（6344 面板，CL 位级相同）；`.vspaero` 里写 `NCPU` 与环境变量 `OMP_NUM_THREADS` 都无效（exe 里没有 `NCPU` 字符串，它用 `Usage: vspaero [options] <FileName>` / ` -omp <N> Use 'N' processes.`）。上表 52×120 档的 212 s 是单线程数，`-omp 16` 后 44 s。
- **差异几乎全在零升力角**：三种方法的升力线斜率 5.743 / 5.576 / 5.205 /rad（差 10%），而 α₀ = −1.95° / −2.32° / −2.05°（差 0.37°）。α = 5° 时 0.37° 只值 ~2.4%，但在 α = 0° 处就是 CL(0) 0.195 vs 0.226（15%）—— **比较小迎角升力时，弯度/尾迹的建模差别会被放大成很大百分比**。
- **`lod` 的 ΣClΔy 系统性低于 polar 的 `CLwtot`**：panel 0.6994 vs 0.7126（−1.8%）、VLM 0.6324 vs 0.6408（−1.3%），差值稳定，说明条带载荷与积分升力是两套求积，不能混用。
- **阻力仍不可用**：panel 模式 `CDwtot` 0.0216 反推 e = 0.69；DPanel Trefftz 反推 e = 1.13（> Munk 极限 1）；VLM CDi 反推 e = 0.62。三者的阻力都落在物理区间外，只能比趋势（§5.2）。
- **`vspapi` 与 DEP-OPT 的几何路径等价**：DEP-OPT 的 `create_wing` 建出的模型用 3.55 剖分，与 `vspapi` 的 `WingSpec` 剖分**输出字节相同**（`runs/vsp_conv/pth_355_re/trap.vspgeom` 与 `api_355_panel/trap.vspgeom` 的 SHA-256 同为 `D7E13757EE8355A92724BB130DA1D12C7E331B039A34301D1E37C222B49CBB9B`），所以 §4.6 / §4.7 的差异既不是模型也不是移植造成的，纯粹是 mesher 版本。

## 5. 归因与陷阱

### 5.1 Trefftz 归一化 bug（已修）

`runs/trefftz_drag.py:12` 原签名 `trefftz_cd(stations, gamma, span, chord, U, rho)` 返回 $D/(q \cdot span \cdot chord)$。对 taper 0.5 的梯形翼，$span \cdot MAC = 2 \times 0.155556 = 0.31111 \ne S_{ref} = 0.30$，于是 CD_T 系统性低 3.7%。修法：加 `area=None`，`sref = span*chord if area is None else float(area)`；`runs/vsp_conv/dpanel_case.py:106` 传 `area=ref_area` 并额外记录 `span_efficiency`。旧数据 0.009056…0.009909 全部按 ×1.0370 更正为 0.009391…0.010276。

解析锚点（`elliptic_test(n=201, b=4, c=1, cl=0.6)`）：$C_D$ = 0.0285025 vs 理论 0.0286479，比值 **0.994925** → 积分实现本身可信（0.5% 是 201 条带的离散误差）。

### 5.2 两侧展向效率都在物理区间外

$e = C_L^2/(\pi AR)/C_D$：

| DPanel 档 | e（用 CL_p） | e（用 CL_Γ） | | VSPAERO 档 | e |
| --- | --- | --- | --- | --- | --- |
| 0.20 | 1.013 | 1.032 | | l1 | 0.946 |
| 0.14 | 1.067 | 1.052 | | l2 | 0.860 |
| 0.10 | 1.112 | 1.055 | | l3 | 0.706 |
| 0.07 | 1.128 | 1.055 | | l4 | 0.615 |
| 0.05 | 1.125 | 1.055 | | | |

平面翼的 Munk 定理要求 $e \le 1$。DPanel 偏大 5%~13%，是因为 Trefftz 积分只用 Γ（环量）而 CL 用表面压力积分，**两条升力不一致**：`CL_p - CL_Γ` 在 0.05 档为 0.695979 − 0.675226 = +3.1%（`docs/ERROR_SEPARATION.md` 记录过同一预算）。VSPAERO 偏小到 0.615，说明它的 CDi 随加密继续增长不是物理收敛，而是**细网格上诱导阻力的离散化误差在积累**（尾迹设置已排除，见 §1 第 5 条）。

→ 所以：$C_{D,DP}/C_{D,VLM} = 0.644$ 这个比值只能当「DPanel 的 Trefftz 阻力比 VLM 的 CDi 低三分之一」来读，**不能当物理结论**；可靠的只有「两侧的阻力都比 5% 收敛准则更差」。

### 5.3 VSPAERO 的 CDo 加性项

| 档 | CDo(α=0) | CDi(α=0) | CDo(α=5°) | CDwtot(α=5°) |
| --- | --- | --- | --- | --- |
| l1 | 0.02046 | 0.00080 | 0.02278 | 0.03073 |
| l4 | 0.02049 | 0.00137 | 0.02279 | 0.03136 |

无粘 VLM 的 CDo 应为 0。这里它是一个与网格、迎角几乎无关的 0.0205 常数，`CDtot = CDo + CDi` 因此被系统性抬高（l4 5° 时 CDtot 0.0387 vs CDi 0.0160）。**任何用 `CDtot` 或 `L/D` 做的对比都要废弃**。

### 5.4 网格非位可重复

0.14 档重跑时壁面数从 1820 变成 1822、CL_p 0.655609 → 0.656163（**+0.08%**）、CD_T 0.009617 → 0.009637（+0.2%）——同 size 下 Gmsh 的三角化不位级可重复（issue #42 记录过坐标最大差 0.1436 弦）。所以本文只能声明到「计数 + 审计」级，网格噪声地板约 0.1%。

### 5.5 为什么 DPanel 的 CL 高 8.6%

三个候选，量级从大到小：
1. **展向自由度（16 个 hat 基 vs 240 条带）**：DPanel 的中展 cl 高 7.8%，而它的总 CL 高 8.6%——两者同量级，说明主要差异来自展向载荷分辨率而不是压力积分。粗展向分布让 DPanel 的载荷更「方」（更接近均匀），同样的总环量下诱导下洗更小、CL 更高。
2. **厚面元的前缘吸力/厚度法向力**：`CL_p - CL_Γ` = +3.1%，VLM 薄片没有这一项。
3. **压力恢复的仿射模板误差**：`CD_p/CD_T` = 0.10，且 CD_p 在 α=0 为负（−0.02），与 issue #49 Phase 0 / PR #29/#30 的结论一致。

## 6. 复现

```powershell
# DPanel 五档（cwd 与 PYTHONPATH 见 runs/vsp_conv/dpanel_case.py 头部）
& .conda_env\python.exe runs\vsp_conv\dpanel_case.py --out runs\vsp_conv\dp_s005 --size 0.05 `
    --alphas 0 2.5 5 7.5 10
# VSPAERO 四档（vsppytools 环境，DEP-OPT 路径）
& C:\Users\zyx20\anaconda3\envs\vsppytools\python.exe runs\vsp_conv\dep_vsp.py `
    --out runs\vsp_conv\vlm_l4 --chordwise 104 --spanwise 240 `
    --alpha 0 2.5 5 7.5 10
# 尾迹敏感性
& C:\Users\zyx20\anaconda3\envs\vsppytools\python.exe runs\vsp_conv\dep_vsp.py `
    --out runs\vsp_conv\vlm_l3w --chordwise 52 --spanwise 120 `
    --alpha 0 2.5 5 7.5 10 --wake-iters 20 --wake-nodes 64 --far 20
# 汇总表 + 图
& .conda_env\python.exe runs\vsp_conv\analyze.py
& .conda_env\python.exe runs\vsp_conv\figures.py
```

DPanel 侧必须 `$env:PYTHONPATH = runs\pr49_git;<ROOT>\audit\runtime\python`、`TMP = TEMP = TMPDIR = runs\pipwork`、cwd = `runs/pr49_git`。

厚面元（panel）模式阶梯 + 聚类二分（vsppytools 环境）：

同一 mesher（OpenVSP 3.55）的 panel / VLM 阶梯与三方法图：

```powershell
# 四档 × 两模式，几何与求解都走 vspapi（需 vsppytools 环境）
foreach ($mode in 'panel','vlm') {
  foreach ($l in @(@('24x036',24,36),@('34x052',34,52),@('44x076',44,76),@('52x120',52,120))) {
    & C:\Users\zyx20\anaconda3\envs\vsppytools\python.exe runs\vsp_conv\api_ladder.py `
        --tag $l[0] --tess-w $l[1] --sect-tess-u $l[2] --mode $mode --alphas 0,2.5,5 --ncpu 8
  }
}
& .conda_env\python.exe runs\vsp_conv\fig22_three_way.py
```

3.52 网格族（§4.6 的历史数据）：

```powershell
# 四档厚网格 + 5 迎角（几何由 dep_vsp --thick --check-only 生成）
& C:\Users\zyx20\anaconda3\envs\vsppytools\python.exe runs\vsp_conv\panel_ladder.py `
    --iters 5 --nodes 8 --far -1 --relax 1
# 根因二分：只换 TECluster/LECluster/MaxGrowth，其余不变
& C:\Users\zyx20\anaconda3\envs\vsppytools\python.exe runs\vsp_conv\cluster_probe.py
# 三方法对照图
& .conda_env\python.exe runs\vsp_conv\fig21_panel.py
```

## 7. 边界

- `physical_accuracy_accepted = false`：没有外部锚点（VSPAERO 同 CAD 的厚面元解、VSPAERO 官方算例、实验数据）来判定谁对，8.6% 的 CL 差和 36% 的阻力差都只是「两码之间」。
- 尾迹几何没有直接对比（VSPAERO 只导出壁面 zone）；DPanel 的尾迹是冻结的，不能拿来验证卷起。
- 粘性、压缩性、配平、舵面一律没有。
- VSPAERO 的 CDtot/L/D 不可用（§5.3）；DPanel 的近场 CD_p 不是良定义泛函——它在 $\alpha = 0$ 给 −0.0207 … −0.0123，$\alpha = 5°$ 随加密从 0.006818 单调降到 0.001031（`CD_p/CD_T = 0.100`），与 issue #49 Phase 0、PR #29/#30 的结论一致，只报 Trefftz 阻力。
- DPanel 只到 12290 壁面（目标密度的 2.0 倍边长），VSPAERO 只到 12480 面（目标密度的 4.0 倍面数）——**两者都没有真正到达用户 traps 网格的弦向密度**，目标密度下的 DPanel 需要约 3 万面元 + 47 GB，超出本机可分配上限 14.45 GB。
- 网格非位可重复（§5.4）意味着重跑会有 0.1% 级差异。