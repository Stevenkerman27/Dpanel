# 连续 P1 G2 —— 耦合主翼与上置平尾

基于 P1 G1（`research/p1-g1-te-pairing-fix@4051ee8`）的研究实现。
**这是真正的第二个独立升力面与尾迹**，而不是分别计算的
机翼载荷之和。旧的 `panel/kernel.py`、P0 生产求解器、
现有 G1 源码与所有 CAD 输入均未改变。

**状态（2026-11-03）**：这条研究线的演示代码保留，`panel/p1_lifting_g2.py` 现在是它的验收判据；基座 P1 G1 已退役（见 [RETIRED_RESEARCH.md](RETIRED_RESEARCH.md)），文中 `research/p1-g1-te-pairing-fix@4051ee8` 在浅克隆里可能检不出，可用归档 tag `research-p1-line-archive` 取回。

## 方法 / 实现

`panel/p1_lifting_g2.py` 为所有升力部件的每个湿面三角形构建
一个共享的 P1 偶极子边界积分系统：

    [D_wall  W_main  W_tail] [mu; gamma_main; gamma_tail] = -phi_inf

每个连通的升力部件上有一个连续的壁面偶极子势场，此外对每个声明的物理
尾迹有独立的 TE 迹自由度分裂。耦合是真实的，因为 `D_wall` 包含跨部件影响，
且 `W_main` 与 `W_tail` 都作用于每个部件的壁面。

每个升力组有自己的展向站位、P1 Gamma hat 基、三角化
物理尾迹、TE 迹跳变、翼尖 Gamma=0，以及独立的 Kutta 条件：

    Jump_g: mu_up_g - mu_down_g = H_g gamma_g
    Kutta_g: signed downstream tangential velocity equality (linear), OR
             Cp_upper - Cp_lower = 0 (nonlinear, analytic Jacobian)

TE 跳变约束由零空间参数化**精确强制**；它不是惩罚残差。
线性 Kutta 模式也是零空间精确约束。在压力 Kutta 模式下，
Kutta 条件是带显式独立于壁面 BIE 的损失项的非线性最小二乘残差。

参考约定：速度量级 1、无粘、绕 y 轴均匀 alpha 旋转，
无 beta，力系数由三角形面 Cp 与法向给出。
`Sref` 默认为**主翼与平尾参考平面形面积之和，仅用于
自包含演示**。对于真实飞机公布的主翼 Sref，使用 `--reference-area 4.0`
（或正确的实际值），以便 CL/CM 比较具有共同归一化。

## 直接运行（无需 CAD/Gmsh）

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m panel.run_p1_lifting_g2 \
      --alpha 0 5 --kutta pressure --out runs/g2_generated_new

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m panel.run_p1_lifting_g2 \
      --alpha 5 --kutta signed_speed --out runs/g2_linear_new

**可选的实际 DPanel 半翼/机身 CAD：**在以修正后的 G1 `research/p1-g1-te-pairing-fix` 分支为基础的工作树中，该分支的适配器使用
`upper_te_faces`/`lower_te_faces` 而不是 TE 边折线作为面元组：

    python -m panel.run_p1_lifting_g2 \
      --main-half-surface runs/cases/wingbody/half_cad_wingbody_v1/surface.json \
      --tail-span 1.8 --tail-chord .55 --tail-x 1.6 --tail-z .35 \
      --reference-area 4.0 --alpha 0 5 --kutta pressure \
      --out runs/g2_real_wingbody_new

CAD 输入保持逐字节不变；它被镜像为完整物理壁面，并在指定
位置加入一个生成的**独立平尾**。这是研究几何，不是经认证的
飞机 CAD 装配。原始主翼无需重新生成 Gmsh/OCC 网格。

`report.json` 包含逐组 CL_p、CL_gamma、Gamma 数组、求解与 Kutta
残差。`mesh.npz` 与每个 `alpha_*.npz` 包含全部实际坐标、
面、湿壁 P1 节点解、逐组尾迹几何、Gamma 与 Cp。

## 实际沙箱计算

机器：Python 3.13.5、NumPy 2.3.5、SciPy 1.17 CPU x64。
固定样例：弯度主翼（展长 4、弦长 1、4x6 弦向-展向面元），
对称上置平尾（展长 1.8、弦长 0.55、4x4 面元）位于
`(x,y,z)=(1.6,0,0.35)`，每组 2 个顺流向面元的尾迹。
192 个真实湿壁三角面元；组合 P1 偶极子影响矩阵
192x108，两个尾迹块分别有 5 与 3 个独立 gamma 自由度（共 8 个）。
两个尾迹面片位于不同 z，因此几何上不是同一个面片。
演示用共同 Sref 为 4.99（=4 + 0.99）。

|alpha|Kutta|CL_p 总量|CL_gamma 总量|CL_p 主翼|CL_p 平尾|壁面 RMS|跳变 RMS|
|---:|---|---:|---:|---:|---:|---:|---:|
|0°|pressure|0.083623|0.092742|0.098794|-0.015171|0.001734|7.9e-17|
|5°|pressure|0.365472|0.342333|0.360551|+0.004921|0.001918|1.4e-16|
|5°|signed speed|0.373501|0.350698|0.3698|+0.0037|—|<1e-12|

**实测干扰，而非叠加：** 在 5° 下分别求解两副机翼，
用相同的总 Sref 归一化：
主翼 CL_p=0.360955，平尾 CL_p=0.056555，和为 0.417510。
**耦合** CL_p 为 0.365472；真实的机体/尾迹影响将
平尾载荷降至 0.004921。各部件有独立的 Gamma 向量，
另一部件从每个尾迹块都看到非零影响。这是
耦合下洗的一个例子，不是经过验证的真实飞机预测。

四项聚焦研究测试通过，包括单组 G1 复现（到
机器精度）、G2 非对角尾迹耦合、实际非线性 Kutta
求解，以及固定几何 alpha 敏感性对有限差分。

## 已知数值/物理限制（不是工作阻塞项）

- 压力 CL 与 Gamma CL 仍然不同（G2 5°：0.36547 对 0.34233）。
- 粗糙近场压力 CD 可为负；无法表示真实总 CD。
- 目前所有尾迹都是固定平面；平尾尾迹只采样规定的
  位置，而不是自由尾迹卷起。
- 最终需要 JAX 几何 VJP、可扩展求解，以及广义独立
  G2 真实 CAD 脱落组输入。本实现是 NumPy/SciPy CPU。
- 真实 396 半翼 CAD 在沙箱中不可用。修正后的
  `from_dpanel_half_surface` 适配器已接线，但这个特定的完整 CAD
  + 生成平尾场景**未在**沙箱中**运行**。
- VSPAERO 不是真值 oracle。实际物理验收仍为 false；
  研究实验应继续，而不是等待门禁扫描。
