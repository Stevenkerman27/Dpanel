# 独立低速面元内核研究

当前通过的是封闭非升力体参考验证。固定尾迹升力表示仍处于失败诊断阶段，不能用于机翼、翼身组合体的设计载荷。

代码按读取网格、几何检查、影响系数、求解、速度、压力、积分和残差顺序执行。依赖 NumPy、SciPy、Numba、threadpoolctl 和 Matplotlib；本机环境为 `C:/Users/zyx20/anaconda3/python.exe`，NCPU 默认 8。

```powershell
& C:/Users/zyx20/anaconda3/python.exe core/reference_cases.py
& C:/Users/zyx20/anaconda3/python.exe core/test_kernel.py
& C:/Users/zyx20/anaconda3/python.exe core/test_kernel_near_field.py
& C:/Users/zyx20/anaconda3/python.exe core/run_references.py
& C:/Users/zyx20/anaconda3/python.exe core/run_wing_diagnostics.py
& C:/Users/zyx20/anaconda3/python.exe core/report_results.py
```

`run_references.py` 在门禁失败时返回非零值。`solve.py` 是诊断工具，退出 0 仅表示计算与输出完成；物理边界检查见 `result.json` 的 `boundary_gates`，完整参考验收见 `summary.json` 的 `gates`。固定尾迹实验不授予“求解器已验证”状态。

## 方法

取 `G=1/(4*pi*|x-y|)`，单层 `S sigma = integral G sigma dA`，双层 `D mu = integral (dG/dn_y) mu dA`，法向朝外。该势函数符号下正源系数对应向内速度。封闭体总势为 `Phi=U dot x + S sigma`，解外侧迹的 `n dot grad(Phi)=0`。常强度三角面采用解析固体角与边对数公式，自面外侧源法向跃变为 `-sigma/2`。不使用涡核膨胀、对角修补、压力裁剪、梯度平滑或快数学近似。

求解前检查正面积、重复面、两流形闭合边、一致朝向与各连通壳体正体积。当前没有实现任意三角面自交和互相穿透检查；通过这些拓扑检查不等于完整几何有效性认证。

`A sigma=b` 由 LAPACK 枢轴 LU 求解并估计 1 范数条件数。闭合连通体分别施加 `C sigma=0`，其中 `C` 是该连通体归一化面面积。默认求解目标明确为 `min ||W^(1/2)(A sigma-b)||^2`，`W_ii=A_i/mean(A_i)`，同时满足零通量约束。通过 `A^-T C^T`、`A^-1 W^-1 A^-T C^T` 和小约束系统实现，不构造大矩阵正规方程。`--mass report` 保留原始配点解，只报告通量。

质量约束有物理意义，但会改变离散配点目标。它产生的真实法向残差必须同时过门禁，不能以“守恒了”为理由忽略无穿透误差。所有速度以单位来流归一化，输出源积分按输入坐标面积换回；`Cp=1-|V_total|^2`，保留法向速度。总力、分部件力和面力均来自同一个 `-Cp*n*A/Sref`。

## 验收与结果

预设门禁和椭球参考推导在 [ACCEPTANCE.md](ACCEPTANCE.md)，实测表在 [RESULTS.md](RESULTS.md)。参考 `Cp` 在平面三角形面心径向投影到光滑球/椭球后评价，比较中包含平面几何近似误差。球、椭球的 L2/L3/L4 为 128/512/2048 面，另有缩放、平移和几何/来流同步旋转验证。面内独立点使用三个不同的重心坐标，不能以配点代数残差代替。

`triangle_kernel.py` 的势和梯度先与独立高阶面积积分、势差分、法向跃变及封闭面 Gauss 恒等式核对，结果在 `kernel_tests.json`。`run_references.py` 记录输入和内核源码 SHA256、归一化坐标、CPU、残差及各项门禁。结果路径为 `results/closed_v1`；`v1` 表示实验批次，不是成熟度版本。

另在距平面和边缘 `1e-2` 至 `1e-12` 的范围检查外侧迹极限及边缘对数梯度。每逼近边缘一个数量级，切向梯度增量收敛到解析的 `log(10)/(2*pi)`；核函数保留该奇异行为。结果见 `near_field_tests.json`。这些测试证明本组核评价没有通过裁剪隐藏理论边缘奇异性，不等于证明任意翼身连接处的全局解都正确。

## 失败的升力实验

`import_vspgeom.py` 读取 OpenVSP v3 原生导出的三角划分、部件标签和尾缘线。原生三角网格与 VSPAERO 的多边形求解网格不同，不能声称是同一离散格式的直接性能对照。

第一版尝试以封闭体源单层加尾迹双层表示升力，并用尾缘相邻上下三角形面心的顺流切向速度相等闭合。尾迹沿来流固定伸展，仅支持直线展向尾缘。它是待证伪的研究表示，不是已验证的通用 Hess-Smith 实现。尾迹积分所得环量载荷仅为独立诊断，不与压力积分相加，符号与物理等价性仍须独立核实。

10 组实验扫描两级网格、迎角 -5/0/+5、质量约束开关及尾迹 10/50/200 弦长。2150 面厚翼、+5 度、50 弦长时，约束后的真实法向 RMS 为 `2.29e-3`，超过 `1e-6` 门禁；不约束时配点误差回到机器精度，但归一化源通量为 `-2.30e-3`，超过 `1e-12` 门禁。Kutta 残差约 `1e-14` 并未阻止异常载荷，压力 CL 为约 `-0.0313`。尾迹长度变化几乎不改变此结果。

这些实验否决当前离散表示，尚不足以唯一定位到核函数、环量符号、尾缘采样或表示空间中的某一个环节。下一步需要在二维已知环量解上独立核实尾迹跳跃、环量符号与尾缘极限，并推导带尾迹的源/双元边界积分表示；若采用势梯度恢复，须单独验证其一致性与真实边界残差。不得通过反转载荷符号、删除质量约束、增大涡核、平滑或裁剪压力让它“通过”。

翼身连接问题暂不进入“已解决”范围。只有升力参考通过后，才有资格将连接处结果用于鲁棒性结论。
