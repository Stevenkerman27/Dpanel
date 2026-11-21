# 求解模型定义

本文给出 DPanel 低速主线**求解的对象、离散形式和可选项的数学定义**。状态数字只认
[STATUS.md](../STATUS.md)；实现位置与接口见 [INTERFACE.md](INTERFACE.md)；验证数据与
参照值见 [VALIDATION.md](VALIDATION.md)；缺陷记录见 [DEFECTS.md](DEFECTS.md)。

## 1 适用范围

| 项 | 定义 |
|---|---|
| 介质 | 不可压、无粘、密度 ρ=1 |
| 流动 | 定常、无旋（势流） |
| 几何 | 三角面元壁面（升力面 + 非升力湿壁机身）+ 尾迹面；P1 线性单元 |
| 尾迹 | 默认固定几何（求解期间不变）；自由尾迹为迭代扩展（§8） |
| 后缘 | 尖锐后缘，Kutta 条件 Δp_TE = 0 |
| 不覆盖 | 粘性、可压缩、非定常、附面层/分离、失速、跨声速、地面效应 |
| 其它物理路线 | 全势/P2（`potential/`）、升力线（`lifting3d/`）、JAX 面元（`core/`）为独立模块，不在本定义内 |

## 2 符号

| 符号 | 含义 | 维度/位置 |
|---|---|---|
| α | 迎角（度） | 输入 |
| U | 归一化来流方向 (cos α, 0, sin α)，|U|=1 | `panel/p1_solve_torch.py:108` |
| x, y, z | 流向、展向、法向 | 全局坐标 |
| μ | 壁面顶点偶极强度（= 壁面总速度势） | n_μ = 壁面顶点数 |
| Γ | 尾迹展向站位环量 | n_Γ = Σ 各 group 站位数 |
| A | 壁面偶极对壁面的影响矩阵（双层位势） | n_μ × n_μ |
| W | 尾迹偶极对壁面的影响矩阵 | n_μ × n_Γ |
| b | 来流项：壁面测点处 −U·x | n_μ |
| M | 壁面 P1 质量矩阵 | n_μ × n_μ |
| J | 势跳变约束 [C | −H]：c 行 μ_up−μ_down = H Γ | n_TE × (n_μ+n_Γ) |
| K | 后缘速度差算子（库塔罚项用的线性部分） | n_TE × (n_μ+n_Γ) |
| p | 后缘压力差向量（逐站 ΔCp） | n_TE |
| G | 逐面 P1 梯度算子：v = G μ | n_face × 3 × n_μ |
| A_ref, c_ref | 参考面积、参考弦长（半模时已镜像为全模） | 标量 |

## 3 连续形式

不可压定常势流：扰动势满足 Laplace 方程，壁面不可穿透、尾迹面法向速度连续且压力连续、
尖锐后缘满足 Kutta 条件（后缘上下表面压力相等）。DPanel 采用**偶极子（Dirichlet，内势为零）**
形式：壁面上的总速度势由偶极分布表示，壁面条件写成第二类 Fredholm 方程；尾迹面上的势跳变
等于该处环量，环量大小由 Kutta 条件确定。

离散后得到线性系统与后缘约束：

    A μ + W Γ = b            (壁面方程；Galerkin 弱形式或配点)
    C μ − H Γ = 0            (尾迹势跳变：上下壁面 μ 差 = 站位环量)

其中 H 是后缘站位 Γ 的线性（hat）插值，两端翼尖处 Γ 固定为 0。

**定义（闭合）**：以上两式只确定"Γ 是势跳变"，不决定 Γ 的数值。把 Γ 定下来的附加条件称
后缘闭合。DPanel 实现两种，见 §5。

## 4 离散

| 项 | 定义 | 实现 |
|---|---|---|
| 壁面未知量 | 顶点偶极 μ，P1（线性）基，每顶点 1 个自由度 | `panel/p1_lifting_g1.py` 网格层 |
| 尾迹未知量 | 每展向站位 1 个 Γ，hat 基，端点 Γ=0 | group.gamma_hat / gamma_y |
| 多组装配 | `combine_wings` 把多片升力面并成一组多 group 网格，Γ 按 group 分块 | `panel/p1_lifting_g2.py` |
| 壁面方程（默认） | Galerkin 弱形式：min ‖A μ + W Γ − b‖² 在 P1 质量内积下，即 M⁻¹ 加权；实现为 M 的 Cholesky 白化 | `panel/p1_galerkin_wall.py` |
| 壁面方程（对照） | 配点：每个面心 1 行，按面积开方加权 | `panel/p1_lifting_g2.py:190` `assemble_multi` |
| 影响系数 | 源三角形 P1 双层位势解析核；目标积分用 Duffy 变换张量 Gauss 求积（默认 order=3）；目标面分批 | `panel/p1_galerkin_wall.py:15` |
| 质量矩阵 | 精确装配 area/12·[[2,1,1],[1,2,1],[1,1,2]] | `panel/p1_galerkin_wall.py:31` |

## 5 后缘闭合的两种定义

| | pressure（默认） | potential_jump |
|---|---|---|
| 数学定义 | 令 s=(μ,Γ)。min ‖B s − b_w‖² + w²‖p(s)‖²，w=5；约束 J s = 0 通过零空间降维精确满足 | J s = 0 精确成立；求 Γ 使 p(Γ) = 0（每站 ΔCp_TE = 0），μ 对 Γ 仿射，内层解一次最小二乘 |
| 附加条件形式 | 罚项（软约束） | 方程（硬约束） |
| 解的唯一性 | 超定最小二乘，无"整条展向 Γ 不变"的近零模态 | 方阵 + Newton；存在多个根（粗网格上不同初值收敛到不同根） |
| 收敛判据 | `pressure_kutta_acceptance`（`panel/p1_lifting_g2.py:269`）：相对梯度 ≤1e-6、wall_rms ≤ 1.05×下界、pressure_rms ≤ 1e-4 | 无专用接受判据，只看 Newton 收敛标志 |
| 失败行为 | 返回状态字段（status/success/message/nfev） | 抛 `potential-jump Kutta...`；未 pinned 时自动回退 pressure 并记录 kutta_requested/effective/fallback |
| 默认值来源 | `PRODUCTION_KUTTA = 'pressure'`（`panel/p1_lifting_g2.py:330`） | 需显式 `kutta="potential_jump"` |

两种闭合的差别只在"把 Δp_TE=0 塞进离散方程的方式"，因此其差异与壁面离散（Galerkin/配点）
和速度恢复位置（§6）耦合。现役校准入口见 [VALIDATION.md](VALIDATION.md)。

## 6 后缘速度恢复

后缘压力差用同一个后缘节点处的上下岸速度计算。

| 恢复 | 位置定义 | 算子定义 | 实现 |
|---|---|---|---|
| `area`（默认） | 上下岸分别取该节点的同岸入射一面环 | 面积加权：w_f = A_f / Σ A_f，v = Σ_f w_f (G_f μ)，只取同 face_bank 的入射面 | `matched_te_velocity_maps(method="area")`（`panel/p1_te_recovery.py`） |
| `quadratic` | 同一个后缘节点 | 1,u,v,u²,uv,v² 六项 QR 最小二乘；秩比 > 1e-8 接受，否则扩环至 5 环；亏秩报错 | 同上，`method="quadratic"`；研究对照，非生产选项 |

两种恢复都作用于同一组 P1 物理自由度，按岸处理并排除端盖。压力定义：

    Cp = 1 − |v|²,   v = G μ

`area` 先在同一个后缘节点分别恢复上下岸速度，再比较两岸恢复速度的平方。
两种 Kutta 闭合都使用恢复后的速度计算后缘残差。壁面载荷由求解后的逐面 Cp 积分。
旧 `face` 恢复仅用于冻结回归及历史对照。

## 7 力与积分量

| 量 | 定义 | 实现 |
|---|---|---|
| 面力 | f_f = −Cp_f · A_f · n_f | `panel/p1_solve_torch.py:174` |
| 合力/力矩 | F = Σ f_f；M = Σ (x_f − x_ref) × f_f | :175-177 |
| CL | F·L / A_ref，L = (−sin α, 0, cos α) | :178-181 |
| CD_pressure | F·U / A_ref（压力阻力，含诱导与型阻的压力部分） | :203 |
| CM_y | M_y / (A_ref · c_ref) | :204 |
| CL_γ | 2 ∫ Γ dy / A_ref（Kutta–Joukowski，逐 group 求和；半模系数 2） | :186-202 |
| CDi（Trefftz） | 精确片线积分：Di = −ρ/(4π) Σᵢⱼ mᵢmⱼJᵢⱼ，Jᵢⱼ = F(bᵢ−aⱼ) − F(aᵢ−aⱼ) − F(bᵢ−bⱼ) + F(aᵢ−bⱼ)，F(t)=t²(log|t|/2−3/4)，端点 Γ=0；CDi = Di/(½ρU²A_ref) | `panel/trefftz_energy_exact.py:31` |
| 展向效率 | e = CL_γ²/(π·AR·CDi) | `tools/wake_closure_study.py:94` |
| 环量锯齿 | Σ|ΔΓ| = Σᵢ|Γᵢ₊₁ − Γᵢ|；变号数 = ΔΓ 符号变化次数 | 验证脚本内联 |

## 8 可微链定义

| 项 | 定义 |
|---|---|
| 设计变量 | 固定拓扑下的几何参数：CST 系数；GRASS 截面参数（展长/弦长/后掠/上反/扭转/厚度/弯度/站数）；或导入面的 (x,z) 坐标 |
| 前向 | 求解一次，并把平稳性残差记入 meta（`_FixedWakeImplicit.forward` `panel/p1_cst_gradient.py:324`，`_PotentialJumpImplicit.forward` :456） |
| 反向 | 隐式函数定理：F(x,θ)=0 ⇒ dx/dθ = −(∂F/∂x)⁻¹ ∂F/∂θ。`pressure` 复用精确 Hessian 的 Cholesky，`potential_jump` 复用闭合 Jacobian 转置的 LU。多载荷 VJP 共享几何算子的重建和求解因子 |
| 边界 | 顶点级 VJP 未实现；接触约束不支持（`fixed_wake_surface` 在存在 contact_pairs 时抛错，:230-231）；拓扑必须固定 |

## 9 约定与不变量

1. 归一化：|U|=1、ρ=1；力以 ½ρU²A_ref 无量纲化，故 CL = F·L/A_ref。
2. 半模镜像：参考面积、参考弦长按全模给出；CL_γ 的系数 2 即镜像贡献。
3. Γ 在两端翼尖为 0；每条尾迹的 Γ 独立，不跨 group 混合。
4. 壁面方程与后缘约束在同一组算子上求解；不同时使用两种闭合。
5. 所有解的输出恒带 `physical_accuracy_accepted = False`（:218）：本模型未通过气动精度验收，
   验证范围见 [VALIDATION.md](VALIDATION.md)。
6. `solution_status` 的判据是灾变门，不是精度门（[NUMERICS.md](NUMERICS.md) §6）。
