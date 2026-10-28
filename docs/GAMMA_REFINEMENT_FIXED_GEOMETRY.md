# 固定几何的嵌套 Gamma 空间实验

关联 Issue #33；基线 PR26 `664d9381686d8cfe1070debba8f9a0e327cbdc51`。
这不是默认后端或精度认证。新增研究模块，不修改生产核、压力、CAD、训练。

## 为什么做这个实验

PR32 本地 `6b84275` 显示 r=0.2Cref 球外仍承担约66%–100%的两恢复算子净阻力差。
它否定“只修翼梢小球”的充分性，不证明差异来自Gamma或尾迹，也不排除全后缘恢复偏差。
以往壁面396→4371的系列多数仍是26站位、22个Gamma自由度，不能代替独立的Gamma空间研究。
新的实验只改变后一个离散因素。不能冻结旧mu后宣称获得新空间的解。

## 不变量与唯一改变量

原wall顶点、面片、法向、面积、源强、壁面偶极块、右端、压力模板/公式、TE端点与上下bank、
来流/密度/参考值、尾迹支撑形状/长度均不变。原接触平台的节点仍共享同一个未知量，
根部自由而非零，翼尖Gamma为零。仅在原站位区间内作1/2/4等分以增加P1环量试验/检验空间。
例如原22 DOF且有22个非平台区间时，理论上得到22/44/88 DOF；以实际report为准。

`split_wake` 对原平面三角形按新站位平面作几何裁分，生成解析积分子片。
这不是调用Gmsh、改wake形状或细化wall。TE原线段也在站位交点处分段，两点Gauss对每段P1基的
乘积精确。原上下bank沿子段继承，不发明新wall状态，不按近邻重新识别TE。
子片没有可微拓扑声明；本研究路径不提供新的几何VJP。

## 方程与必要恒等式

原/新空间都解同一形式：

```
D*mu + W*g = b
-K*mu + M*g = 0
K = integral_TE h(s) * wall_jump_incidence(s) ds
M = integral_TE h(s) * h(s)^T ds
```

求解前按原算法的 `measure=integral h ds` 缩放后两块。M的Cholesky仅用于检查正定；
求解采用现有SciPy稠密LU，不作伪逆、截断或正则化。零支持DOF/秩亏/原方程真残差失败均拒绝。

设P把旧自由Gamma值延拓成新自由Gamma值，应满足（M、K均在行缩放前比较）：

```
W_fine @ P = W_old
P.T @ M_fine @ P = M_old
P.T @ K_fine = K_old
```

这些恒等式来自旧分片线性函数被嵌套新空间精确表示，以及相同物理支撑上的积分可加性。
它们不要求不同空间的新解相等，也不证明物理精度。1x必须先重现原JAX半模A各块、mu/gamma、
原Cp/CL/CD/CM。新增空间后重新求解mu/gamma，并复用完全不变的 `_half_loads`。

## 实现范围

- `panel/gamma_refinement.py`：接触相容P1空间、延拓P、平面三角形/条带裁分、TE分段积分、
  分块调用原 `triangle_potential`、嵌套恒等式与联合LU。
- `panel/run_gamma_refinement.py`：原生半模基线一次求解、1/2/4实验、逐级报告/NPZ、拒绝覆盖、
  原surface SHA和源码SHA、资源门限、失败报告。
- `factor_*.npz`：mu/gamma/sigma、站位/DOF、原始Cp及面力矩、细分wake积分三角形、parent、
  每子片实际常量Gamma与y斜率。保留全表面，不删尖峰。
- `te_jump_l2_over_Uc` 是离散mu上下跳跃投影误差，不是TE压力连续性或完整物理Kutta验收。
- 理想Trefftz统一在每个原区间32等分的相同网格上求积，防止native站位数量变化混入估计器误差；
  它仍是另一个物理模型，不作优化目标。可另用64等分检查这个诊断量的求积敏感性。

`half_panels<=1500`、子片<=20000、每核块target*source<=1000000、理想诊断全站位<=4001
是显式研究预算，不是普适硬件能力声明。超过先拒绝；本地可以减小block或有根据地提高门限。
仍存半模稠密D/LU，不能声称矩阵无关或GPU加速。没有新依赖。

## 当前已运行的证据

本次云端 Python3.13.5 / NumPy2.3.5 / SciPy1.17.0 / JAX0.9.0.1 CPU Float64：

```
python -m pytest -q test/test_gamma_refinement.py
23 passed (约4秒，无skip)
```

包括接触平台/自由根/零尖、嵌套场精确表示、每原三角形面积/一阶矩守恒、精确TE质量矩阵、
反向TE绕序、接触Galerkin恒等式、无TE支持拒绝、尺度/平移、非法/资源拒绝、非对称联合LU、
实际解析wake核的嵌入/镜像、独立32x32面积Gauss积分对照。使用原核完整源码；其git blob
`93d8f429ae9587f2a01db078c9fdf49d7ef02390`与远端一致。

当前环境无法git直连（DNS不可达），只获取了所需原kernel及新模块；没有完整Dpanel checkout。
`test_gamma_refinement_integration.py` 的两个真实仓库测试、整个CLI与真实396/1205翼身
尚未在云端执行。不能将上述23项数学/核测试称为完整系统验收。

## 本地执行

独立worktree，沿用现有JAX/SciPy/NumPy，不改生产环境。先运行：

```powershell
python -m pytest -q test/test_gamma_refinement.py test/test_gamma_refinement_integration.py
python -m panel.run_gamma_refinement --surface cases/panel_half/thick_wing_two_strip.json --out runs/gamma_small_v1
```

临时目录权限失败可使用此前等价tmp_path插件，但须分别报告原命令/环境例外；skip不是通过。
仅小夹具1x回归和1/2/4嵌套检查通过后：

```powershell
python -m panel.run_gamma_refinement --surface <原396面surface绝对路径> --out runs/gamma_396_v1 --block 16
python -m panel.run_gamma_refinement --surface <原1205面surface绝对路径> --out runs/gamma_1205_v1 --block 16
```

每档独立进程，外部记录峰值RSS；不重跑CAD。1966面需单独评估并显式`--max-half-panels 2200`，
不作为首轮必跑。输入应为PR29所用原fixture，不能换网格造成混杂。

回报HEAD/版本/exit/首traceback；实际DOF与子片数；1x所有回归差；W/M/K嵌套残差；
真实系统及weak Kutta残差；三档CL/CD/CM和原始Cp极值/位置、组件ID、Gamma/斜率；耗时/RSS。
新增Gamma状态不是PR29的冻结原状态，不能冒用旧state hash；也不能把新细分wake NPZ直接当
`half_planar_cut.v1` fixture送入旧加载器。PR30后续可用保存的直接+镜像子片场做独立双侧审计，
需要明确适配，不能伪称已兼容。

## 判读与停止条件

1x不复现或嵌套恒等式失败：实现/版本问题，先停，不分析物理。
1→2→4若Gamma斜率产生振荡或CD/CM更敏感：如实保留，可能揭示固定wall跳跃的限制，不调参数
追Trefftz。若逐项变化小，结论仅是这套固定几何/固定压力恢复对Gamma空间较不敏感，不能
关闭尾迹运动学、全后缘恢复或外部基准工作。

生产继续 `physical_accuracy_accepted=false`，无正式训练、无默认替换。任何精度改善声明仍需
同几何独立参考与wall/Gamma/wake分离收敛；本PR不会因残差小而改变物理门禁。

## 方法背景与来源边界

- Erickson, *Panel Methods: An Introduction*, NASA-TP-2995 (1990)：源/偶极子、边界条件、跳跃和
  尾迹建模的公开背景：https://ntrs.nasa.gov/citations/19910009745 。
- Maskew, *Program VSAERO Theory Document*, NASA-CR-4023 (1987)：常源/偶极子、内部Dirichlet
  及尾迹形状迭代的背景：https://ntrs.nasa.gov/citations/19900004884 。

以上不声称文献直接实现本实验。嵌套P、分片积分及恒等式为针对仓库现有weak Kutta的代数推导；
实际核、镜像斜率符号、归一化、压力恢复均来自已锁定Dpanel版本，不借用外部经验修正。
