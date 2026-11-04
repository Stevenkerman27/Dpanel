# 连续 P1 G1：全机翼三角面元、实际尾迹与独立 Kutta 求解原型

**Status:** research prototype; `physical_accuracy_accepted=false`。本代码是可以求出三维机翼 `mu_node`, `Gamma(y)`, `Cp(face)`, `CL`, `CM_y`, `CD_pressure` 的端到端研究算例，不是 GRASS 生产后端，也没有靠 VSPAERO 调参。旧 P0/G0 源码不改。

## 为什么这次不同于旧的 G1 实验

旧 G1 联立 `D mu+W Gamma=-phi_inf` 和 `mu_upper-mu_lower-Gamma=0`，但后一式只**定义势跳跃**，不能选定环量，实际 396 面翼身的环量升力接近零。本原型在其上增加**独立的后缘速度闭合条件**，用真正的机翼表面切向速度选定环量。

定义 `mu_h|T = Σ_a mu_a N_a`，上、下翼面光滑连接节点共享 `mu`；在尖后缘的上、下势迹**仅分裂势自由度**，几何节点不移动。尾迹由流向多个三角条带组成，双层势密度 `Gamma(y)` 使用固定展向 P1 hat 函数，沿流向传输；末端两个翼尖 Γ=0。完整三角形 P1 解析双层核沿用 PR #57 `p1_closed_body.influence_block` 的等价源点一阶矩公式。源项使用 G0 已验证的纯双层 Dirichlet 形式（不叠加 P0 常源层）。

求解约束：

```
         D_wall mu + W_wake Gamma = -phi_inf           (壁面势方程，过定最小二乘)
         (mu_upper - mu_lower) - H_te Gamma = 0       (TE 势跳跃定义，精确约束)
         K(mu,flow,te_geometry) = 0                    (独立物理 Kutta 选择)
```

`K` 有两种模式：
- `signed_speed`：在后缘附近的上、下表面匹配**沿弦向的有符号切向速度**，是线性的工程近似，和势跳跃共同作为**精确线性等式约束**。通过其 null-space 上的 QR/SVD 最小二乘求解，没有人为惩罚权重。
- `pressure`：匹配上、下 TE 附近平均切向**速度平方**（等价于同一来流条件下的压力相等），是非线性边界条件，以前一方案作初值。TE 势跳跃始终**精确**；压力条件通过带有解析 Jacobian 的非线性最小二乘逼近，残差原样输出。

这是对有限数目 TE 近场采样的闭合模型，不宣称等同于真实钝后缘的分离／基底压力物理。**对称翼 0° 不产生人为 Γ；正迎角由方程自行生成 Γ，不在代码里指定目标升力。**

## 可运行的两条端到端路径

**生成三维封闭尖后缘翼**（NACA00XX 厚度、余弦弦向分布、完整 span、翼尖封口；可加光滑弯度）：

```shell
python -m panel.run_p1_lifting_g1 --chord-panels 6 --span-panels 8 --camber 0.02 \
       --alpha 0 5 --kutta pressure --out runs/p1_g1_cambered
```

**从原有 DPanel 正半域 CAD 三角形网格导入研究原型**（按 `faces/wake_faces/trailing_edges/gamma_stations` 重建完整镜像，不运行 Gmsh）：

```shell
python -m panel.run_p1_lifting_g1 --half-surface runs/half_cad_wingbody_v1/surface.json \
       --alpha 0 5 --kutta pressure --out runs/p1_g1_real_cad
```

后一入口已经用沙盒生成的半翼 `surface.json` 完成了**实际完整运行**，但尚未拿到用户本机原始翼身 STEP/396 面 `surface.json`，所以不能声称用户的实际 CAD 已通过该适配器。对于真钝后缘，目前保留上下侧物理节点及基底，Kutta 是**实验性近似**；每个 TE bank 通过真实 `trailing_edges` 索引与邻近面元几何确定，不是任意 CAD 零件都附会涡片。

输出目录包含 `report.json`（标量/残差/γ）、`mesh.npz`（所有壁面/尾迹/DOF拓扑）和每迎角 `alpha_*.npz`（真实求出的 μ、Γ、逐面 Cp），不覆盖已有目录。

## 已在本次沙盒真实执行

环境：Python / NumPy 2.3.5 / SciPy 1.17（CPU）；不需要 Gmsh；所有数值来自上述求解代码直接运行，**不是设计目标或手写的参考升力**。

| 研究模型 | 壁面三角形 | 工况 | Kutta | CL_pressure | CL_gamma | wall RMS | jump RMS |
|---|---:|---:|---|---:|---:|---:|---:|
| 对称闭后缘翼，弦4/展6划分 | 112 | 0° | signed_speed | 0.000000 | 0.000000 | 0.00180 | <1e-14 |
| 同上 | 112 | 5° | signed_speed | 0.321120 | 0.283773 | 0.00213 | <1e-14 |
| 2% 弯度翼，弦6/展8划分 | 216 | 0° | pressure | 0.141862 | 0.147539 | 0.00156 | <1e-14 |
| 同上 | 216 | 5° | pressure | 0.388562 | 0.379741 | 0.00178 | <1e-14 |
| 独立正半翼 JSON 镜像重建 | 80 | 0° | pressure | ~0 | ~0 | 0.00195 | <1e-14 |
| 同上 | 80 | 5° | pressure | 0.285109 | 0.227607 | 0.00235 | <1e-14 |

- `signed_speed` 的 **CL 对 α 的固定几何解析导数**，4/6 网格 α=5°：`dCL/dα_rad = 3.63495870359`，独立两侧有限差分为 `3.63495870249`，相对差约 `3.0e-10`；尚不是几何形状梯度。
- 测试（精简为四条高信息量研究测试）：**4 passed**，包括正迎角实际生成 Γ、线性／非线性 Kutta 一致、解析迎角导数、半翼 JSON 导入全场求解。
- `Cp` 积分与 `Gamma` 升力仍有差距；较粗网格的 `CD_pressure` 可能为负。这些是研究结果，不通过裁剪或选择输出掩盖，也不能用于工程设计优化。
- 不同三角形剖分方式可能影响此粗网格结果；正半翼镜像与独立完整翼的三角剖分对角线不同，因此不要求同值。

## 下一步应做的研究，而非增加验收清单

直接利用此 G1 计算框架扩充：对真实已有翼身 `surface.json` 做一次端到端求解，读取 `CLp/CLGamma/Cp/CM` 和各后缘局部速度；依据实际异常修订 Kutta/截面采样，而不是先执行厚度五档。随后将 NumPy 的固定拓扑算子迁移到 JAX x64 可微路径，使用已有 PR #57 G0 几何导数体系。推广 G2 时每个真实翼/尾翼独立 Γ 和尾迹组。

**模型不完美是已知研究事实；保留原始 P0 默认，研究的 G1 原型可持续运行和修改。**
