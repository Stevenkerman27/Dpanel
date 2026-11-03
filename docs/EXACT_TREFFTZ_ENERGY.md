# 分段线性 Gamma 的解析理想 Trefftz 积分（研究）

Issue #49 / PR #54 后续工作。这是一项**只读的数值
积分修正**，而非新的环量、Kutta 条件、尾迹动力学，
也不是生产阻力。旧版 `ideal_trefftz_piecewise_linear`
在分段中点采样诱导下洗。它得到的数值
**会在分段线性 Gamma 被细分而函数本身不变时发生变化**。
那是积分误差，不是新的物理模型。

当两个物理翼尖均为 Gamma=0 时，分部积分给出能量

```text
w(y) = -PV integral Gamma'(s)/(y-s) ds / (4*pi)
D_i = -rho integral Gamma(y)*w(y) dy
    = -rho/(4*pi) sum_ij m_i*m_j*J_ij
m_i = (Gamma(b_i)-Gamma(a_i))/(b_i-a_i)
J_ij = integral_[a_i,b_i] integral_[a_j,b_j] log|y-s| ds dy
     = F(b_i-a_j)-F(a_i-a_j)-F(b_i-b_j)+F(a_i-b_j)
F(t) = 0.5*t*t*log|t| - 0.75*t*t; F(0)=0
```

因此，对于*固定的分段线性 Gamma*，能量可以
解析求值，而无需站点中点下洗积分。
新模块 `panel/trefftz_energy_exact.py` 在显式的
**全展长** Gamma 上实现该式，两端翼尖为零，分段数有界，
并使用分块的 O(N²) 跨段矩阵。调用代码必须自行重建
物理上一致的镜像站点值。源手性、全展长、参考面积
与远尾迹假设的校验仍由调用方负责。

## 实际执行过的独立测试

`python -m pytest -q -p no:cacheprovider test/test_trefftz_energy_exact.py`
=> **16 passed / 0.15 s**，且源文件/测试文件的 Git blob 与
实际执行的文件一致。测试包括：
- 在 [-1,1] 上 Gamma(y)=1-|y|，Sref=2、U=rho=1：
  精确 CDi=log(2)/pi 与 CL=1。
- 对**同一**分段线性 Gamma 做 2/4/8/16 倍细分，
  必须把 CL 与 CDi 保持到浮点容差。
- 密度/速度/Gamma/坐标/参考量的缩放与平移、
  无环量、非法翼尖值、重复 y 与输入预算。
- 独立的冻结 12290 壁面梯形模型，5deg、Gamma16，
  取自 `docs/issue49_vspaero/gamma/dp_s005_a5/gamma_profile.json`，
  位于 `research/issue49-vspaero-evidence@d24e4db`。

## 同一环量，七种积分结果

| 原始区间细分数（旧中点法） | CDi_ideal |
|---:|---:|
| 1 | 0.0102758545335 |
| 2 | 0.0108529738957 |
| 4 | 0.0111186303578 |
| 8 | 0.0112416745197 |
| 16 | 0.0112996369585 |
| 32 | 0.0113274327229 |
| 64 | 0.0113409580368 |
| **解析对数能量** | **0.0113541664454** |

原始 DPanel `dpanel.json` 中使用的粗网格值
**比同一 Gamma 的这次精确积分低 10.49%**。PR #34/#54
对每个区间使用 32 次细分，报告 CDi=0.01132743；它
与解析值相差约 0.24%，而 Gamma 自由度从 16 变到 64 只让
共网格结果变化约 0.04%。

此前基于 Gamma CL 与粗网格 CDi=0.01027585 的展长效率估计
约为 e=1.059（对于平面椭圆理想基准，>1）；
在 Gamma 固定并使用解析 CDi 时，该值约为
e=0.959。因此，表面上的 e>1 **不能单独用来
诊断非物理的环量/尾迹**。使用*压力* CL 而非
Gamma CL 仍会得到不同的 e，两种力估计量
不可混为一谈。

## 硬性限制

该解析表达式针对分段线性全展长 Gamma，积分的是所选定的
**半无限平直尾迹理想升力线模型**。
它不是有限的 DPanel 尾迹动量计算、不是黏性阻力
预测、不是自由尾迹解，也与 VSPAERO `CDiw` 的离散力通道
不同。它不确立 DPanel 的物理阻力
精度，也不意味着任意非平面机翼或
多尾迹状态能被该理想估计器准确建模。

在新程序于 NACA/椭圆解析载荷、VSPAERO 固定尾迹
配置和多个真实 Gamma 分辨率上通过检查之前，
不要把它设为生产默认。每条历史数值记录都应
以明确的勘误说明保留，而不是覆盖它。
不改网格，不做 GRASS 训练。
`physical_accuracy_accepted=false`。
