# 连续三角 P1：真正的 G0 闭合体求解（研究实现）

关联 Issue #49、#56，基于 PR #52 的解析 P1 双层矩核。**不修改原 kernel.py、half_native.py、尾迹、生产默认或 GRASS 训练。**

## 1. 交付的不是插值器

给定一个完整、外法向一致的闭合凸三角曲面和来流 U，未知量是湿壁面节点上的扰动势/偶极强度 mu。每个面上的 mu_h=sum(N_a*mu_a)，相邻光滑面使用同一节点值。

采用与原 G0 P0 基线一致的表示：

    sigma_f = -n_f dot U
    D^- mu_h = S sigma
    phi_ext(x) = U dot x + D mu_h(x) - S sigma(x)

S 是逐片常源解析积分，D 是真正按三角形线性基函数积分的双层势。求解过程不读取球体解析势，不使用旧 P0 mu 插值；解析解只出现在 benchmark 中。

## 2. 节点未知量不要求节点配点

Issue #56 证实逐片平面扇角和不能充当整个三维角点的自由项。本实现**不使用那条规则**，也不试图给棱边/顶点硬填 -1/2。

试验空间是连续 P1；测试点位于严格的面内（质心，或每面3个内部重心点）。每个自面局部迹使用 -N_a(x)/2。面元数大于连续节点数，所以系统天然是矩形：

    minimize_mu || diag(sqrt(A_f/q)) (A mu - S sigma) ||_2

用非截断 QR 求解，并对 R 的奇异值做秩检查；不解法方程、不使用正则化/伪逆、不补虚拟方程。归一化权重只乘公共面积常数，不改变最小化问题。这是**面积加权超定配点研究离散**，不是宣称完成了 Galerkin 或通用节点 collocation。

1280面球体：1280方程、642节点未知量。矩阵满列秩，常数模式 A*1=-1（不是零空间），**不固定 gauge**。BIE 残差与 QR 一阶最优条件残差分开报告；前者不应被宣传为机器零。

支持单一、闭合、外绕向、凸、genus-zero G0 物体。输入检查边闭合/绕向、凸性、退化/重复面、实际壁面节点压缩和预算。未引用坐标不变成未知量；不自动合并物理割面。半模、开放翼面、Kutta/尾迹、多体及非凸输入尚未支持。

## 3. 文件与入口

- panel/p1_closed_body.py：NumPy 真正连续 P1 装配、QR 求解、P0 控制、表面载荷及独立外场。
- panel/p1_closed_jax.py：同一离散的 JAX x64 小型可微求解；几何、来源、面积权重、QR、压力链均可求导。**在 NumPy 拓扑/秩验证后使用，固定合法拓扑，最多320面。** 几何退化不可继续求导。
- panel/run_p1_closed.py：同网格 P0/P1 解析球端到端命令，固定空间探针，拒绝覆盖旧结果，失败写 failure.json。
- test/test_p1_closed_body.py、test/test_p1_closed_jax.py：31项实际测试。

NumPy 默认最多2560面/1400节点、800万矩阵元素；JAX求解上限320面是参考实现预算。独立外场 AD 可对更大已求得的 NumPy 状态采样，不代表通用性能保证。

## 4. 本次实际测试

Python3.13.5、NumPy2.3.5、JAX0.9.0.1、CPU/float64，OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。完整 Git clone 因容器 DNS 不可用；新模块在隔离目录执行，P0核是已核对的原文件，Git blob **93d8f429ae9587f2a01db078c9fdf49d7ef02390** 与远端逐字相同。**不是完整仓库回归。**

最终：**31 passed，0 skipped，12.68s**。包括与原 JAX 核对照、P1矩系数独立36×36 Duffy求积、非质心面内迹、共享节点矩形真求解、制造RHS、满列秩/常数模式、孤立节点/非法网格/预算、变换与尺度、CLI和禁止覆盖、JAX/NumPy一致性、完整几何方向导数和压力目标梯度对照。

端到端：80/320/1280面，各自从来流重新求 P0、P1；同一三角几何、同一P0源。固定96个Fibonacci方向+6轴向点，共102个点。外场速度是**对完整势函数做JAX空间自动微分**，不做切向投影；参考为球体解析势流，不是VSPAERO。

### Cp RMS，在相同外部点 r=a*(1+offset)

| 面/节点 | P0 0.01a | P1 0.01a | P0 0.02a | P1 0.02a | P0 0.20a | P1 0.20a |
|---|---:|---:|---:|---:|---:|---:|
|80/42|3.413053|0.176022|1.217604|0.145959|0.064209|0.059378|
|320/162|1.549158|0.074880|0.603890|0.053732|0.015362|0.015843|
|1280/642|0.745627|0.028694|0.292077|0.015959|0.003798|0.004051|

近壁改善显著，但**远场并非全档改善**。指标包含多面体与光滑球的几何误差，不能将改善倍数外推到机翼/后缘。

### 负面结果必须保留

独立外场在面心外侧 0.001*sqrt(face_area) 的真实法向速度 RMS，除以 U=1：

| 壁面 | P0 | P1 |
|---|---:|---:|
|80|0.037306|0.063779|
|320|0.021946|0.034821|
|1280|0.014512|0.018736|

P1 此指标随加密下降，但**比同网格P0更差**。不能用显式投影后的表面速度伪造无穿透通过。独立三配点/面对照也已运行：1280面时近壁0.01a Cp RMS=0.028547，法向RMS=0.019143，没有解决缺陷。

1280面P1加权BIE残差约5.29e-4，QR最优性残差约4.10e-17。真实节点未知量已求出，**G0整体物理验收仍未通过，不能推广至生产或G1**。

完整摘要及源文件SHA见 P1_CLOSED_RUN_RECEIPT.json。

## 5. 本地复测：独立 worktree，新目录

    python -m pytest -q -p no:cacheprovider test/test_p1_closed_body.py test/test_p1_closed_jax.py
    python -m panel.run_p1_closed --levels 1 2 3 --samples 96 --derivative jax --out runs/p1_closed_NEW
    python -m panel.run_p1_closed --levels 1 2 3 --samples 96 --derivative jax --samples-per-face 3 --out runs/p1_closed_q3_NEW
    python -m panel.run_p1_closed --levels 1 2 --samples 32 --derivative fd4 --out runs/p1_closed_fd_NEW

PowerShell 先设置 $env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'。失败保留首异常，不重跑CAD，不改网格，不将未执行的全库回归算进通过数。

先复跑新测试和原PR52/原解析球；再评价固定角向探针增加至384的敏感性、同离散几何外迹偏移扫描、面积加权BIE残差及其导数。下一阶段优先降低真实外迹法向残差（可研究严格弱式测试/相容源表示），**不直接进入Kutta升力面或多尾迹**，不以贴近VSPAERO数值为目标。

## 6. 来源与边界

- Bempp Handbook: https://bempp.com/handbook/api/function_spaces.html 与 https://bempp.com/handbook/api/boundary_operators.html 。参考P0/P1空间和单双层定义，不调用Bempp。
- Maierhofer & Huybrechs, An analysis of least-squares oversampled collocation methods for compactly perturbed boundary integral equations in two dimensions, arXiv:2201.11683, DOI 10.1016/j.cam.2022.114500。公开摘要提供超定配点背景，**其二维理论不证明本三维离散的稳定性/收敛阶**。本文依赖实测秩、残差、球体对照。

physical_accuracy_accepted=false。可运行的连续P1 G0研究求解及小型可微链已交付；任意CAD/非凸体、角点节点配点、Kutta机翼、自由尾迹、GPU性能与GRASS物理验收尚未完成。
