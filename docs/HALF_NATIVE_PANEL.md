# 右半侧输入的原生镜像面元法（half-native）

> 状态：**数值研究接口与小型严格对称厚翼验证；非生产整机、非已验收物理精度。**
> 本开发堆叠在 `feature/matrix-free-gmres-symmetry`（PR #17）之上，不修改默认 `panel/solve.py`、Torch 桥、CAD、压力算法或 GRASS 训练。Dpanel Issue #20 的 **真实 OCC 正半域剖分** 尚未实现。

## 动机与数学约束

此前 PR #17 半模接口以完整镜像网格为输入，导致旧的全机独立 Gmsh 三角网格因缺少镜像节点/面元而被 `make_plan` 拒绝。正确的生产路线是：先建立真实外露的 **y>=0 翼身公共外表面与流体侧尾迹**，只对该半域做一次共形剖分；求解器仅接收该半域，镜像侧作为潜在作用的**瞬时积分几何**，无独立物面 μ/环量 γ 自由度。

本 PR 首先完成求解器侧，不引入不可靠的通用 CAD 半域布尔：

- `panel/half_native.py`：只接收正半侧的顶点、壁面三角形、wake、真正上/下 TE 面元及**非负**循环站位。将反射的面元三角顶点反序，使法向保持外向/尾迹 +z；在右侧配点计算 `S_direct+S_image`、`D_direct+D_image`、`W_direct+W_image`。镜像的线性展向斜率在 y 反射下变号，所以 `My_direct - My_image`；这是避免遗漏的关键。
- `prepare_half_topology`：离散、只执行一次的共形性审核。用半模型生成**O(N) 的虚拟面片连接关系**仅用于邻域表，检查镜像后无裸边、有相反边绕序、正体积、物理 TE 两 bank 均真实与 wake 接边。不持久化镜像坐标或镜像状态变量；镜像 ghost 仅以 0..2N-1 的索引访问压力重构中心。
- `solve_half_fields`：小型隐式可微 **半模稠密 LU 参考**。只分解 `(N_half+G_half)^2` 规模的矩阵，仍使用 Dpanel 原解析源/双层势积分与 `linear_solve`，返回半侧 raw Cp / 面力、**全飞机**压力力与力矩及局部/参考系数。CG 参考点须在 y=0。
- `HalfPanelOperator`：可对同一半域的流式 `Av`、严格解析 `Aᵀv` 和 RHS 分块求值，不形成完整半域或全域 AIC；每次矩阵作用仍 O(N²)，工作块 O(block·N)，不是 FMM。
- `panel/half_native_gmres.py`：CPU SciPy GMRES，局部偶极子+镜像小块 LU 与半 Kutta 下三角预条件，并提供 P⁻¹/P⁻ᵀ、GMRES 正向和转置求解。必须按**真实残差**拒绝不收敛。该 host wrapper 只是求解参考，**尚未接 JAX custom VJP**。
- `panel/run_half_native.py`：实验 CLI，从只含正半侧的 `dpanel.half_planar_cut.v1` 读取网格；按 lu/gmres 分开输出结果、真实残差和计时，原有目录一律不覆盖。

### 变量、对称、单位

输入 `coordinates:[V+,3]` 的 y>=0；`faces:[F+,3]` 与 `wake_faces:[W+,3]` 都只在右侧。`gamma_stations:[S+]` 从 y=0 开始严格递增；`gamma_station_dof` 的外侧 tip=-1 表示 γ=0。真正左右两侧的镜像贡献在同一积分核内计算，`half μ` 无须对应独立左侧 μ。

物面源项对纵向对称来流满足镜像偶对称。上下 TE 在 y 反射下仍是 **upper→upper / lower→lower**（不是上下对调）。全机力、力矩由半侧分别按极向量 R=(1,-1,1) 与轴向量 `det(R)R=(-1,1,-1)` 积分：`F_full=F_half+R F_half`，`M_full=M_half+det(R)R M_half`。力矩参考点必须在 y=0。输出采用 SI（N、N m），`reference_area` 是**完整飞机**参考面积，不得用半侧湿表面积替代。

仅支持：对称几何与设计方向、零侧滑、固定正交平面尾迹、实际且正确的 TE 两 bank、固定拓扑及合法一阶变形。不能对接触拓扑跃迁、非对称副翼/侧滑、自由尾迹或 OCC 重网格自动求导。

## 数值验收

新 `cases/panel_half/thick_wing_two_strip.json` 使用 **右侧 20 个封闭厚翼物面三角形 + 4 个流体侧尾迹三角形 + 2 个 γ 未知量**，没有左侧三角、左侧独立 μ/γ，也没有中心面封盖。

`test/test_half_native_panel.py` 在测试内部**临时构造**镜像全模作为独立 oracle（完整 40 壁面 + 8 尾迹，43 个状态未知量）。全模矩阵折叠 `R A E` 与直接半模矩阵比较，并检查 state、raw Cp、全机压力力/力矩；通过 JAX VJP 与三档 FD 检查半域几何及展向站位；检查分块 `Av/Aᵀv`、转置内积；GMRES+块预条件同时测试正向和转置残差。明确拒绝侧滑、偏心参考点和非闭合壁面。

本次云端隔离环境 JAX 0.9.0.1/Float64、NumPy 2.3.5、SciPy 1.17.0、CPU：直接参考矩阵最大差 **1.39e-17**，raw Cp 差 **1.33e-15**，载荷最大差 **4.44e-16**；联合几何/站位导数 `dJ/dθ=0.00232260857520`，三档差分绝对误差 **4.34e-13 / 5.66e-13 / 3.43e-11**。分块 A/AT 相对稠密乘的向量范数误差在 **5e-16** 内；小例 CPU 近场预条件 GMRES 正向/转置 13 次，原残差约 **5.5e-12/5.9e-12**。这些来自本次隔离同物理核公式的数值测试，**提交后仍须本地使用真实仓库的 `panel.kernel` 与 JAX 0.10.2 重跑**。

运行（单独 worktree，输出目录不能存在）：

```bash
python -m pytest -q test/test_half_native_panel.py

python -m panel.run_half_native \
  --surface cases/panel_half/thick_wing_two_strip.json \
  --out runs/half_native_lu_v1 --mode lu

python -m panel.run_half_native \
  --surface cases/panel_half/thick_wing_two_strip.json \
  --out runs/half_native_gmres_v1 --mode gmres --block 7 --near 6 \
  --tol 1e-10 --restart 40 --cycles 20
```

本地额外要检查：`PANEL_KERNEL_BLOCK=64` 的真实 Dpanel kernel、JAX 0.10.2 的 `lax.scan` 转置规则、Linux/Windows 峰值内存、完整字段差和失败/非法模式。如真实核发生偏差，请回完整 traceback，不要临时截断 Cp 或调大镜像容差。

## 下一件事：真正只剖分半架飞机

见 Dpanel Issue #20。需要**独立的** `half CAD -> conforming right exterior mesh` 管线：

1. 从 OCC fused solids 与 fluid-side wake 构造 y>=0 半域；保留翼身公共交线、实际 TE 和流体侧 contact-cut，半域临时计算用的对称面封盖**不能变成气动 wall 面片**。
2. Gmsh 只为真正外露的右侧壁面/尾迹各生成一次网格。根部与中面使用稳定拓扑接缝 ID，不允许跨平面三角或重合的重复 seam 节点。输出本 PR 的 `dpanel.half_planar_cut.v1`，并在生成阶段复核 wake/TE bank、组件分区、邻接、原始压力模板。
3. 在同一个半网格上先运行 `solve_half_fields`，再检查半流式 `HalfPanelOperator`/GMRES；只在**测试时**镜像生成全模并与 full LU 的 state/Cp/forces/moments/VJP 比较。旧的左右独立 Gmsh 网格是不同离散，不可逐面强行对拍。
4. 所有通过后再把 CPU GMRES 的显式 `Aᵀ` 与 `jax.custom_vjp` 接成完全可微迭代的生产候选，最后才接 GRASS CST/机身几何生成器。初期禁止越过 `physical_accuracy_accepted=false` 的物理精度门槛。

**本次代码没有修改 GRASS 或现有 Dpanel 生产 solve/训练入口，没有真实机身 CAD 半域网格或 GPU 吞吐验收。**
