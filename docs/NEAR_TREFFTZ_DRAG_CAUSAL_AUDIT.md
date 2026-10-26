# Dpanel 半模翼身：近场压力阻力与理想 Trefftz 阻力的因果归因

2026-10-26。只读研究，不改变已经验收的 JAX、Torch、OCC 或压力公式。本报告依据本地 21abb754 提交的 MESH_CONVERGENCE.md，ERROR_SEPARATION.md 和 PR10_PRESSURE_RECONSTRUCTION.md。所有数值仍保留 physical_accuracy_accepted=false。

## 1. 八档网格的真正结论

同一 NACA4412 翼身、5°迎角、右半壁面 396→4371，LU 相对残差约 1e-15：

| 右侧壁面 | CL（Cp） | CL（Gamma） | CD（Cp近场） | CDi（理想Trefftz） | 近/远 | CM_y |
|---:|---:|---:|---:|---:|---:|---:|
| 396 | 0.531230 | 0.545601 | 0.006168 | 0.022978 | 26.8% | -0.022267 |
| 805 | 0.567183 | 0.577672 | 0.008915 | 0.025667 | 34.7% | -0.018291 |
| 1205 | 0.580904 | 0.589946 | 0.009252 | 0.026824 | 34.5% | -0.018268 |
| 1966 | 0.593817 | 0.595464 | 0.011038 | 0.027357 | 40.3% | -0.019185 |
| 2921 | 0.591273 | 0.593456 | 0.012885 | 0.027399 | 47.0% | -0.016837 |
| 4371 | 0.607135 | 0.609897 | 0.016968 | 0.028645 | 59.2% | -0.019860 |

CL 两种算法正在趋同，最细差约 0.45%。但近场 CD 仍比理想诱导阻力低约 40.8%，CM_y 仍非单调。注意：1205→4371 的理想 Trefftz CDi 仍变动 6.79%，805→4371 累计变动 11.60%。相邻档小于5%并不足以证明严格网格无关，更不意味着全部气动系数已经收敛。

## 2. 已证实的数值敏感性：局部压力梯度 + 巨大相消

近场的原始压力系数来自两列切平面、无加权、面心势差的最小二乘梯度，然后由伯努利关系形成 Cp，对面元法向面积积分并投影到来流。其数学秩可以通过，但不等于真实梯度准确。PR10 的独立仿射 QR 完全复现生产梯度（排除2x2算法Bug），简单距离加权反而产生 Cp -40 级发散，直接二次重构有52/680面元不满足五项秩条件。问题是重构模型和所使用的数据，而不只是线性代数。

396 面元本地真实全机 F=(-0.098381299,0,1.297876942) N，迎角5°、qS=2.45。来流轴向力投影是：
- Fx cos(alpha)/(qS) = -0.04000282796
- Fz sin(alpha)/(qS) = +0.04617037915
- 相减净 CD = +0.00616755118

仅这两个总力分量的绝对和/净值就是13.97倍；逐面正负压差的相消还会增加条件敏感性。这可以解释为什么CL接近却不能保证CD近场收敛：局部Cp、法向、前后缘离散的一点偏差便可导致净阻力很大相对变化。**但相消是误差放大机制，不能凭它断言全部40.8%的缺口都来自Cp。** 本地诊断发现小面积异常Cp和翼身TE细条不是近场净阻力的主要贡献，不应裁剪或删除它们来“修复”结果。

## 3. 实际存在的模型不自洽：固定水平有限尾迹、斜向来流

请直接核对项目：
- panel/cad_half.py 中的 OCC wake points 使用 z=0；
- 同文件却给 U=(U cos(alpha),0,U sin(alpha))，alpha默认5°；
- panel/half_native.py 和 half_surface_mesh.py 对wake要求常数z且面法向+z。

因此固定wake上 freestream 法向通量的无量纲值为 |U dot n|/U=sin5°=0.0871557427，即8.7%。真正定常自由尾迹应满足相对于**总局部速度**的运动学切向条件，远下游应趋于来流方向。这里水平x向尾迹仅是简化规定形状，不能因此声称它与真实远尾迹等价。诱导速度可能抵消局部的部分法向通量，故8.7%不是完整wake穿透残差，更不是已量出的CD误差。

Dpanel JAX求解用**有限长度**的固定三角偶极wake（body、source、wake解析势）；本地 runs/trefftz_drag.py 则由Gamma(y)解析PV条带对数给出**理想半无限直尾涡**下洗。前者还受body、有限wake截断、真实尾迹切割影响；后者不包含这些三维场贡献。两个估计器不是对同一个真实尾迹作完全相同的控制体积分。

本地椭圆Gamma校准得到趋近 CL²/(pi AR)，它验证理想Trefftz条带算法和因子，却不能单独验证实际翼身、root-contact或固定尾迹的物理等价性。严禁用该CDi反向强制校准Cp。

## 4. 壁面和尾迹分辨率没有一起细化

396到4371壁面计算通常固定26个展向站位/22个Gamma自由度。壁面细化11倍，环量基函数和翼身root-contact并未系统同步加密。size=.13时改8/16/32条wake strips，CL(Cp)分别0.580904/0.589125/0.576162，近场CD分别0.009252/0.010063/0.008245，非单调。须做壁面密度×Gamma站位两维独立扫描，另改变wake长度和方向；不得把多种离散混为单一“网格加密”。

## 5. 两种第三方实现的实测源码对照

**VortexAD**:
- 非结构三角后处理 post_processor_tri.py 使用 unstructured_least_squares_velocity_tri 得到切平面势梯度，再按 Cp=1-|V|²/U² 和面片法向积分力；least_squares_velocity.py 同样是局部2x2法方程。因此它可能遇到同类Cp/近场CD敏感性；换VortexAD不会自动消除。源码：
  https://github.com/LSDOlab/VortexAD/blob/main/VortexAD/core/pm/post_processor_tri.py#L120-L205
  https://github.com/LSDOlab/VortexAD/blob/main/VortexAD/core/pm/source_doublet/least_squares_velocity.py
- 上游有Cp_cutoff和平滑下限（部分文档默认-5）；它可以改变局部力，**不是物理收敛修复**，Dpanel仍应输出原始Cp。
- post_processor.py 另有可选Trefftz阻力，trefftz_plane_drag.py 的新实现以**实际固定wake长度的75%位置**抽取诱导速度，包含body source、doublet、wake contributions。这给出了值得仿真的第三条**实际wake中/远控制面**，独立于纯Gamma理想Trefftz。但是该实现存在wake速度core/位移等规定参数，不能直接视为严格无核物理基准。
  https://github.com/LSDOlab/VortexAD/blob/main/VortexAD/core/pm/post_processor.py#L239-L269
  https://github.com/LSDOlab/VortexAD/blob/main/VortexAD/core/pm/trefftz_plane.py#L98-L245
- Dpanel先前对VortexAD 1616面元严格镜像AIC进行块分离，发现其半模不等价性主要来自wake/Kutta修正块而非body source。这说明该类程序的尾迹/TE匹配也很敏感；但不等于VortexAD已经被证明有相同近场CD网格不收敛问题。

**FLOWPanel.jl**:
- FLOWPanel_postprocess.jl 包含 Green–Gauss cell-centered、area-weighted nodal-centered 以及按边界/面法向角度选择的混合梯度；尖TE节点有upper/lower分开处理。值得在相同冻结mu和半网格上做后处理对照。
  https://github.com/byuflowlab/FLOWPanel.jl/blob/master/src/FLOWPanel_postprocess.jl#L124-L960
  https://flow.byu.edu/FLOWPanel.jl/v1.2/geometry/panel-gradient/
- 但其某些Kutta pressure correction是**上下TE Cp取均值**，源码明确对unstructured SimpleMesh警告不正确；另有clip/maxgrad置零选项。这些启发式修补不能直接用于Dpanel原始物理验收。
  https://github.com/byuflowlab/FLOWPanel.jl/blob/master/src/FLOWPanel_postprocess.jl#L1059-L1097
- FLOWPanel有FastMultipole研究入口，但FMM路径含未完成的Kutta/wake TODO；FMM优化计算效率，不会自动提高本轮Cp阻力准确性。

经典气动文献也明确指出：近场压力阻力极易因正负力相消而失准，远场Trefftz相对更稳定；但自由/非平面/复杂wake须独立处理动量守恒，不能简单相信一种估算器。

## 6. 本分支新增可复现实验及下一步因果隔离

新增只读 panel/drag_consistency_audit.py：
1. 读取原有 dpanel.half_planar_cut.v1，默认最多1600右半壁面，调用已验收的JAX半模LU（不重新CAD）。
2. 用原始Cp和实际三角面积向量独立重积分全机CD/CL，并与solver原有force做代数核对。
3. 报告面元正负drag、绝对和/净值的 cancellation factor、Fx与Fz投影、body/wing组件贡献及敏感性预算。
4. 从gamma_station_dof重建镜像偶对称Gamma，以解析PV区间对数和无截断Sref公式计算**理想**Trefftz CDi，不强迫和近场相等。
5. 对真实有限wake报告 |Uinfty dot n|/U（只代表自由来流角，不代表总局部wake穿透）。
6. 记录SHA256、reference、真实数值与physical_accuracy_accepted=false。纯NumPy数学测试已在隔离环境通过（椭圆Gamma/镜像root-contact/压力投影/5°尾迹/坏输入）；项目内JAX小网格端到端需本地再测。

本地命令：

- python -m pytest -q test/test_drag_consistency_audit.py test/test_drag_consistency_jax_integration.py
- python -m panel.drag_consistency_audit --surface ABS_SAVED_PATH/half_cad_wingbody_v1/surface.json --out runs/drag_audit_396_v1.json
- python -m panel.drag_consistency_audit --surface ABS_SAVED_PATH/half_cad_wingbody_3x/surface.json --out runs/drag_audit_1205_v1.json

更大档只有确认内存足够，独立进程显式加 --max-half-panels 6000。输出禁止覆盖旧 runs。

下一轮先完成 **A：同一mu/Gamma只换压力后处理**（原有仿射LS、cut-aware Green–Gauss、upper/lower不共享数值的one-sided nodal），必须先过manufactured gradient、TE跳跃与旋转/尺度守恒测试；无需重新LU。

然后做 **B：wake方向和长度物理实验**。在OCC里用来流倾斜wake并重新fluid-side裁切，调整解析wake核/几何门禁；不能把旧JSON的wake点简单旋转导致穿透实体。比较0/2.5/5°，wake长度10/30/60c，理想Gamma-Trefftz和基于实际body+source+wake速度的固定控制面。每一项均列真残差、无穿透、Kutta/Cp差及近/远场CD。

最后做 **C：wall网格×Gamma分辨率矩阵**，验证CL、近场CD、理想CDi和CM收敛。现阶段近场CD与CM不可用于GRASS训练reward，理想Trefftz CDi只能表述为“该Gamma在理想半无限尾涡模型下的诱导阻力”，不是已验收真实总阻力。不得放松任何物理精度门槛。
