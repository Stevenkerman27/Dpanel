# 可解释 PyTorch Float64 半模有升力面元法（CPU/CUDA 原型）

**状态：独立、可选择启用的研究后端。** PR #22 的 JAX 原生半模与 Gmsh 正半域 CAD 管线、正式 \`panel/solve.py\` 和 GRASS 训练入口均保持原样。该 PR 的价值是可解释的 PyTorch 积分/线性求解/一阶梯度链；既不宣称物理精度通过，也不预先宣称 GPU 比 LU/JAX CPU 更快。

## 算法和边界

\`torch_panel/half_lifting.py\` 不 import JAX，核心计算可以完全驻留在 PyTorch CPU/CUDA 的 Float64 张量上。CAD/mesher 与固定拓扑压力邻居仍在主机端构造和验证，求解器只接收：

\`\`\`python
solve_half_torch(vertices, station_y, topology, freestream,
                 density, reference_area, reference_chord, reference_point,
                 block_size=16, implicit_adjoint=True)
\`\`\`

连续变量包括半侧节点坐标、非负尾迹/环量站位、来流、密度、参考面积/弦长与力矩参考点；所有度量量——面积、法向、源强、尾迹和 TE 基函数、表面速度、原始局部压力载荷及力臂——都在当前 PyTorch 计算图中重新计算。离散量包括连接、上下 TE bank、wake strip、hat map、组件 ID 与带真实接触切边的压力模板，它们是**冻结的拓扑**，不对 Gmsh/OCC 布尔或邻居选择求导。

### 势积分与镜像

- 正侧真实面元的解析常源势 \`S\` 和常偶极子势 \`D\`，面元内部双层自迹固定为 **−1/2**；有限流体侧尾迹使用**解析**偶极子 \`Dw\` 及线性 y 一阶矩 \`My\`。未引入 vortex core、\`Cp\` 截断、奇异值截断、配点移动或数值奇异积分的隐式 epsilon。
- XZ 镜像 \`(x,y,z)→(x,−y,z)\` 反转三角绕序，保持外法向；真实/镜像源与偶极子贡献叠加，镜像线性展向斜率的符号反转：
  \[
  D_h=D_{\mathrm{direct}}+D_{\mathrm{image}},\quad
  W_h=(D_{w,d}+D_{w,i})B+(M_{y,d}-M_{y,i})B'.
  \]
- 当前几何重新计算尾迹 \`B,B'\`、后缘积分、弱 Kutta：
  \[
  A_h=\begin{bmatrix}D_h&W_h\\-T&M\end{bmatrix},\qquad
  A_h[\mu;\gamma]=[S_h\sigma;0].
  \]
  根部 \`gamma\` 不强行设零，接触段通过半域 \`hat_map\` 合并，翼尖 \`gamma\` 为零。
- 沿真实 TE/wing-body wake contact **切断**压力重构邻接；当前 \`panel.pressure_stencil\` 先在冻结几何上按秩预算选合法模板，再用现有二维仿射未加权表面势梯度恢复原始 \`Cp\`。物理力和轴向量力矩通过极/轴向镜像分别合成全机结果，面积参考量使用**整机** \`Sref\`。

### 稠密 LU 与显式隐式伴随

原型使用 GPU 可执行的 \`torch.linalg.lu_factor/lu_solve\` 处理半模增广矩阵。自定义的一阶 \`ImplicitLUSolve\` 不反传 pivot：

\[
A^\mathsf{T}\lambda=\partial J/\partial u,\quad
\partial J/\partial A=-\lambda u^\mathsf{T},\quad
\partial J/\partial b=\lambda.
\]

这样 Torch 的一般自动微分继续将 \`dA\` 和 \`db\` 传给顶点、尾迹、来流等，载荷面积/法向/力臂的直接几何导数也保留。\`implicit_adjoint=False\` 使用普通 \`torch.linalg.solve\` 供独立导数对照。这是**单层一阶 VJP**，不是二阶导数、FMM、无矩阵求解或者多 GPU 实现。

## 数值交叉验收，而不是浮点逐位比对

\`test/test_torch_half_lifting.py\` 与 \`panel/verify_torch_half.py\` 调用同一半域 \`dpanel.half_planar_cut.v1\`，**只在主机端预处理一次切边和压力邻接，两个后端使用完全相同的冻结拓扑**。对照项目包括：

\`matrix_half/rhs_half\`、\`mu/gamma/sigma\`、尾迹 \`B,B'\`、TE 基函数、\`Cp_raw_half\`、法向/面积/切面速度/模板行列式、局部面力和力矩、机翼/机身组件力矩及全机 \`CL,CD_pressure,CM\`。对数值差异用绝对/相对容差，而不是比特一致：CPU 测试默认 \`rtol=2e-6, atol=2e-8\`；CUDA 数值测试 \`rtol=5e-5, atol=1e-7\`；验证 CLI 默认 \`rtol=2e-5, atol=1e-7\`。逐字段报告最大绝对误差、relative L2 和逐元素是否超差。**接受小差异≠允许系统性的载荷或梯度偏差。**

一阶反向还要对照 PyTorch 自身 \`torch.linalg.solve\` 梯度、JAX \`jax.grad\`，以及三档中心差分（1e-3、1e-4、1e-5）。CPU-only 环境的 CUDA 测试会明确 skip；请求 \`--device cuda\` 而没有 GPU 则**硬失败**，不伪装成 CPU 结果。报告包括求解真残差、两端计算时间、显式 CUDA allocated/reserved 峰值和保存场 SHA。任何源文件都不覆盖旧 \`runs\` 输出。

### 本次可执行环境已经实际检查的内容

Torch **2.10.0+cpu** / Python 3.13 / Float64 / 一张与项目测试相同定义的右半厚翼（20 壁面/4 尾迹/2 gamma）：本次隔离运行 \`CL=0.08328459438725033\`、\`CD_pressure=0.0094924275905445\`（与 PR #21 JAX 本地对应数据一致），LU 相对残差约 1e-16，所有局部 Cp 有限。自定义 LU 伴随与 PyTorch 内置 \`torch.linalg.solve\` 返回的顶点/站位梯度最大差约 \`2.0e-15 / 1.4e-16\`。**这些是隔离 CPU 数值检查，不是新建 PR 的整仓测试或真正 CUDA 实测**；完整 JAX 逐字段和真实 396 面翼身对照仍须本地执行。

### 本地执行顺序（PowerShell）

\`\`\`powershell
git fetch origin
git switch feature/torch-half-lifting-cuda-prototype
python -m pytest -q test/test_torch_half_lifting.py

python -m panel.verify_torch_half \`
  --surface cases/panel_half/thick_wing_two_strip.json \`
  --out runs/torch_half_small_cpu_v1.json --device cpu --derivatives

python -m panel.verify_torch_half \`
  --surface <已有绝对路径>/half_cad_wingbody_v1/surface.json \`
  --out runs/torch_half_wingbody_cpu_v1.json --device cpu --derivatives

python -m panel.verify_torch_half \`
  --surface <已有绝对路径>/half_cad_wingbody_v1/surface.json \`
  --out runs/torch_half_wingbody_cuda_v1.json --device cuda --derivatives --block 16
\`\`\`

实际 396 wall + 22 gamma 的翼身半域文件已经过 PR #22 的自适应模板恢复满秩与 JAX LU/GMRES 验收；使用其 SHA **相同的** \`surface.json\`，无需重跑昂贵的 STEP/Gmsh。不要用旧的左右独立 Gmsh 全模直接按面编号对照。

本地应报告每个命令 EXIT、Python/Torch CUDA/JAX 版本和 GPU 型号、all-fields 相对 L2/最大误差/是否超差、Float64 真实残差、CPU 和 GPU 冷热前向/反向墙钟、峰值 RSS 和 GPU allocated/reserved。CPU/真 GPU 两类结果不能混写。

## 性能和物理限制

此阶段仍为**稠密半模 LU**：AIC 存储 \`O(N_half²)\`，LU \`O(N_half³)\`；通过按目标块计算解析核避免物理层 \`O(N²×9)\` 的一次性中间几何张量，但最终矩阵、LU 因子与 autograd graph 都会占用显存。真实 GPU 双精度性能高度依赖具体 NVIDIA 卡；CPU-only 通过不能声称 GPU 更快。

在一张固定网格上可做一阶连续几何优化，但改变翼身接触拓扑、网格剖分或秩自适应邻接选取时，目标函数并不保证光滑。原有压力积分升力、网格收敛和外部物理基准门槛未解决，**\`physical_accuracy_accepted=false\`** 保持，正式 GRASS 飞机生成训练不启动。

要扩展至数千/万面元，还需要 CUDA 常驻的分块 AIC/核融合、更好的显存管理，必要时针对大规模引入有误差预算的 FMM/H² 或低秩压缩，并验证隐式伴随的转置作用；不能只靠把 CPU LU 替换为 GPU LU 得到规模复杂度提升。
