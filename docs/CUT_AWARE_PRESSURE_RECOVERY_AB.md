# 固定势场的分侧节点压力恢复 A/B（研究，不替换生产）

依据本地 `a3270fa` 的 PR26 五档审计。基于 PR26 `664d938`，不修改已有核、半模求解、Kutta、CAD、GPU、训练入口。关联 #27；实际总流尾迹审计另由 #28 执行。

## 判断边界

同一离散 Cp 的重积分闭合约 1e-17，仅排除该积分实现/单位通路差错，不排除曲面离散、面内求积或压力恢复误差。强相消只表示敏感性，不证明某项净阻力是虚假值。部件的负压力阻力不单独证明错误。标签20/30在历史报告中名称矛盾，本工件只使用数值ID，需由CAD来源映射确认。

直接解析源/偶极子场速度与切平面拟合速度不是同一算子。不能要求两种 Cp 逐面相等，也不能把恢复后的切向速度当作独立无穿透证据。近场CD与理想Gamma-only Trefftz来自不同尾迹模型，不通过强制匹配来选择后处理。

## 本次候选

`panel/pressure_recovery_ab.py` 只依赖 NumPy。

1. `affine_face_gradient` 用冻结的原邻接、两列切平面设计矩阵、无权QR重现原算法。它是控制组，不是精度改进本身。
2. `cut_vertex_gradient` 先沿真实 wall 连接建立邻接；在每个顶点，仅以不跨 TE/contact 切边的 incident-face 连通扇作为一个数值节点。不同 bank 的同坐标节点拥有不同势值，不能平均。
3. 每个节点以原始面心 `mu` 作满秩三列仿射拟合 `[1,xi/h,eta/h]`，QR求解；`h` 为几何距离尺度，不是正则化。只在秩比不足时沿 cut-free 图有界扩展，且禁止进入该顶点另一扇。SVD仅报告最小/最大奇异比；不截断，不伪逆。默认最多4次扩展，阈值1e-6；失败给出顶点/扇/深度并退出，不静默换算法。更大跨度仍可能受曲率和远绕切口影响，须检查输出patch，而非把不直接跨cut当无限距离的分侧保证。
4. 从各三角形三个分侧节点势构造P1面梯度：`grad(mu)=sum mu_k [n cross (x_(k+2)-x_(k+1))]/(2A)`。等价于该线性三角形的Green–Gauss边积分。使用与生产相同的表面势约定 `V_t = P_t U_inf + grad_s(mu)`，保留新的原始 `Cp=1-|V_t|²/U²`，无Cp裁剪/平滑/上下压力平均。

这是**新的离散恢复候选**，不是解析场的精确trace；不保证在真实翼身更准，不自动恢复壁面无穿透或自由尾迹压力连续性。NumPy研究路径无VJP；只有物理与收敛证据支持后才单独移植固定patch的JAX/Torch可微实现。

借鉴 FLOWPanel 的 cell→node→triangle-gradient 分层，但本实现采用分侧仿射QR而非其面积平均；没有复制上游源码或TE压力均衡修正。原始方法说明：
https://flow.byu.edu/FLOWPanel.jl/v1.2/geometry/panel-gradient/
该页面的矩阵印刷有重复行，实际实现按三个不同顶点的重心梯度公式推导。

## 冻结与复现

`panel/compare_pressure_recovery.py` 默认仅调用**一次**已验收的 `solve_half_fields`；随即保存 `frozen_state.npz`（mu/gamma/sigma/原Cp/原邻接/真残差/输入SHA/代码来源），后续两种恢复读取完全相同且只读的状态。可用 `--snapshot` 完全绕过JAX求解。固定解的重复后处理不是新物理解。

输出新目录保存：输入副本、冻结状态、逐面候选Cp/梯度/节点势NPZ、每个节点扇的support/秩比/depth、原/控制/候选的分量与总CL/CD/CM及正负相消、模块SHA。失败也保存report，禁止覆盖。构造O(N)镜像几何仅用于诊断后处理，不构建或求解全模AIC。对于没有明确snap到y=0的半模节点，本工具拒绝，不焊点。

本地先完整checkout，不得只复制一个文件到旧local分支：

```powershell
git fetch origin
git switch research/cut-aware-nodal-pressure-recovery
python -m pytest -q test/test_pressure_recovery_ab.py test/test_pressure_recovery_half_integration.py
python -m panel.compare_pressure_recovery --surface <原396面绝对路径>/surface.json --out runs/recovery_396_v1 --max-rings 4
python -m panel.compare_pressure_recovery --surface <原396面绝对路径>/surface.json --snapshot runs/recovery_396_v1/frozen_state.npz --out runs/recovery_396_replay_v1 --max-rings 4
```

依次1205面与1966面（后者显式 `--max-half-panels 2200`），每档独立进程，记录峰值RSS/耗时/版本/HEAD。优先读每档baseline原Cp一致性和候选失败；不要先跑4371或为通过而改变cut、容差或默认压力。

## 已运行与待验收

云端隔离新NumPy模块：Python3.13.5/NumPy2.3.5。十项测试覆盖不规则平面仿射与常数、势规约偏移、TE跳跃及故意删cut的污染、旋转/平移/尺度、镜像、面序/循环顶点序、原正规方程等价、非法输入及秩亏拒绝、曲面制造势加密。

球面制造势 `mu=U*x/2`（面心沿半径映射到单位球后采样，并非运行面元方程）：

| 三角形数 | 面积加权梯度RMSE | Cp RMSE |
|---|---|---|
| 32 | 0.0910856 | 0.278594 |
| 128 | 0.0316167 | 0.0808991 |
| 512 | 0.0125829 | 0.0312586 |

仅证明本制造场上恢复一致性/加密趋势，不是求解器球体或翼身验证。完整半模JAX集成、CLI状态重放与真实翼身A/B**待本地checkout执行**。缺JAX导致skip不计通过；原沙箱tmp_path权限问题若再现需单独标注，不改物理断言。

## 接受候选的条件

先基线原Cp复现，再逐网格比较候选原始Cp、局部载荷和CL/CD/CM趋势，结合#28独立场/尾迹残差及外部同几何基准。净CD更小、为正、接近Trefftz或奇异比改善都不能单独接受。任何秩亏、镜像不一致、新尖峰、细化恶化均记录为候选失败，不以缩放或裁剪修补。外部物理误差未过门槛前保持 `physical_accuracy_accepted=false`；不进入GRASS正式训练。
