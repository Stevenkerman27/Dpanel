# 保守全势方程研究原型

本目录已实现并运行二维圆柱和 Joukowski 升力翼型、完全亚声速的保守全势原型。它用于决定快速中低保真度求解器的路线，尚不是通用翼型或三维翼身求解器。

## 升力阶段（2026-10-23）

`solve_airfoil.py` 在保角映射体网格上使用周期势加独立环量自由度，联立尖后缘 Kutta 条件；不是将解析环量赋给解。二维势分支切割保证环量记账，没有实现三维自由尾迹。

主序列 16 组中，15 组亚声速通过预设门禁，M=0.6 明确超出当前中心椭圆格式范围。M=0、5 度、128x512 的压力 CL=0.58824984，解析 CL=0.58817981，相对误差约 0.0119%。28 项解析力/环量、通量重算与构造一致性检查通过。另完成 4 组尖后缘误差诊断和 12 例独立二维面元参考。

局部尖后缘法向重构误差仍须改善。追加诊断确认径向分辨率和映射奇点的误差放大；三次外推在一档网格上存在误差抵消，未据此改写原始压力或通过结果。详见 [LIFTING_RESULTS.md](LIFTING_RESULTS.md)、[LIFTING_PROTOCOL.md](LIFTING_PROTOCOL.md) 和 [CUSP_PROTOCOL.md](CUSP_PROTOCOL.md)，图为 `results/lifting.png` 与 `.pdf`。

```powershell
& C:/Users/zyx20/anaconda3/python.exe potential/run_lifting_study.py
& C:/Users/zyx20/anaconda3/python.exe potential/audit_lifting_cusp.py
& C:/Users/zyx20/anaconda3/python.exe potential/test_lifting.py
& C:/Users/zyx20/anaconda3/python.exe potential/report_lifting.py
```

`test_lifting.py` 同时读取 `audit/panel_airfoil_hesssmith/n256_ap5` 的独立面元结果，可由 `panel_airfoil.py --panels 256 --alpha 5 --out audit/panel_airfoil_hesssmith/n256_ap5` 重建。两种方法的未知量、几何和外边界不同，不能由当前精度/时间表宣称全势普遍优于面元。

## 方程与实现

以圆柱半径、来流速度、来流密度为单位，`gamma=1.4`，总速度为 `V=grad(Phi)`：

```text
div(rho * grad(Phi)) = 0
beta = 1 + (gamma-1)/2 * M_inf^2 * (1 - |V|^2)
rho = beta^[1/(gamma-1)]
M_local^2 = M_inf^2 * |V|^2 / beta
Cp = 2/[gamma*M_inf^2] * {beta^[gamma/(gamma-1)] - 1}
```

`M_inf=0` 时密度为常数，方程退化为 Laplace，`Cp=1-|V|^2`。小非零 Mach 的压力计算使用 `expm1`，避免两个接近 1 的数相减；这是等价公式的数值评价，不修改压力。

`solve_cylinder.py` 从上到下生成极坐标网格、计算密度、组装共享面通量、求解、计算原非线性残差及压力。单元中心存储总势；内部每个面只生成一个导通系数，分别以相反方向进入两个相邻控制体。内边界质量通量为零，周向周期，外边界固定总势。密度在相邻单元之间取算术平均，外边界系数取最外层单元密度。

每一步冻结密度，调用 SciPy 的稀疏直接求解，再按更新后的势和密度检查非线性质量散度。此处没有实现 AF2、多重网格、Newton、密度上风或激波耗散，也没有从 WBCAero 反编译移植其求解内核。其理论来源是公开的等熵无旋方程；当前网格和离散是明示的研究实现。

`beta<=0` 或迭代中出现局部 Mach 达到 1 时显式退出。这个范围检查针对当前椭圆型中心格式，不能解释为全势方程无法处理局部超声速；跨声速需要另行构造和验证有方向性的守恒离散与激波选择机制。

## 边界误差的解释

有限体积边界的零质量通量通过离散方程施加。同时用前三层单元中心势独立重构壁面法向梯度，并将其保留在压力所用完整速度中。两个量分别报告。

`128x256`、`M=0.3` 的独立法向重构 RMS 仍约 `1.78e-3`，不能声称达到了 `core/ACCEPTANCE.md` 对配点原型要求的 `1e-6` 点态门禁。本目录的预设验收是控制体质量残差、低 Mach 一致性和圆柱解析压力收敛；两种离散目标不相同。本次证明的是受控原型的这些性质，不是翼身边界条件已通过原项目全部要求。

圆柱外边界 `Phi=R*cos(theta)` 的 M=0 解析解为有限环域解，不能与无限域圆柱混比。M>0 本轮没有独立可压缩解析或实验参考，暂不认证其绝对压力精度。

## 复现

```powershell
& C:/Users/zyx20/anaconda3/python.exe potential/run_study.py
& C:/Users/zyx20/anaconda3/python.exe potential/test_models.py
& C:/Users/zyx20/anaconda3/python.exe potential/report_study.py
```

NCPU 请求值为 8；线程库受 threadpoolctl 限制，但 SciPy 稀疏直接求解器不保证使用八线程。计时为当前代码的单次内部测量，不含 Python 启动和最终输出压缩，也不是优化实现的性能上限。

固定实验方案见 [PROTOCOL.md](PROTOCOL.md)，数据表见 [RESULTS.md](RESULTS.md)，机器可读数据为 `results/cylinder_v1/summary.json`，每例另存 `field.npz`、源代码 SHA256、每步残差与退出状态。图为 `results/pilot.png` 和 `pilot.pdf`。

共运行 14 组全势配置：13 组亚声速配置通过本目录预设门禁，1 组 M=0.45 在迭代中触发局部声速范围限制。另有 4 组独立二维面元圆柱参考。独立线积分、定向面通量累加和等熵总焓检查共 24 项，通过记录为 `model_tests.json`。

面元圆柱的均匀多边形面心 Cp 有特殊精确性；它不代表任意几何上的机器精度。面元的无限域边界与本全势有限远场也不同，不能用这张计时表排列两类算法的普遍优劣。

## 后续门槛

二维升力翼型、环量与 Kutta 已完成上述受控验证；下一步改进尖后缘局部重构与分辨率匹配，再建立跨声速实验参考和离散。新加入的上风/耗散必须给出守恒通量、方向判据、参数、激活位置和加密行为。三维阶段须同时验证网格自动化、连接处几何、局部法向误差、质量余额、分部件压力积分、载荷与速度/内存。

有限体积没有让几何退化、窄间隙、小单元或边界插值问题自动消失。三维体网格生成和边界离散是独立的路线决策条件。
