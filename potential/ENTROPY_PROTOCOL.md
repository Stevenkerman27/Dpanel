# 一维 E-O 辅助演化协议

## 目的与范围

本协议验证一个标量辅助守恒律：

```text
q_t + f(q)_x = 0
f(q) = -rho(q) q
rho(q) = [1 - (gamma - 1) M_inf^2 (q^2 - 1) / 2]^(1/(gamma - 1))
```

它是稳态一维全势质量余额的辅助演化系统，用来检查 E-O 通量对压缩/膨胀 Riemann 初值的数值选择。它不是二维全势方程求解器，不认证二维跨声速流、保角网格度量、物面边界或激波后的真实熵增。

参数固定 `gamma=1.4`，`M_inf=0.5, 0.8`。对每个来流马赫数取 `qL_super=1.2 q*`，再用 `brentq` 在 `(0,q*)` 解 `rho(qR) qR = rho(qL) qL`，使通量两侧满足 Rankine-Hugoniot 零速条件。要求整个 `[qR,qL]` 上 `f''(q)>0`。压缩算例为左超声速 `qL`、右亚声速 `qR`；膨胀算例交换左右态，作为不满足熵条件的反向跳跃初值。

## 通量与更新

按原文 E-O 定义，令 `q1` 为通量的第一个参数、`q0` 为第二个参数：

```text
m(q) = rho(q) q
s(q) = 0                         if q <= q*
s(q) = m(q) - rho* q*            if q > q*
h(q1,q0) = -m(q1) + s(q1) - s(q0)
```

在有限体积面上明确调用 `F(q_left,q_right)=h(q_right,q_left)`，故 `h=-massflux` 且参数顺序不可交换。网格为 `[-1,1]`，跳跃放在 `x=0` 的 cell face；网格数 `N=64,128,256,512`。采用显式 Euler：

```text
q_i^(n+1) = q_i^n - (dt/dx) [F_(i+1/2) - F_(i-1/2)]
```

每步取

```text
dt <= 0.4 dx / (max(0, max_[qR,qL] f'(q))
                 + max(0, -min_[qR,qL] f'(q)))
```

最终步缩短到目标时刻，不做状态 clipping、不加额外人工扩散。左、右边界按标量特征速度解释：入射特征施加对应 Riemann 远场态；出射特征使用最近单元外推。目标时刻取 `T=0.25/max(|f'(qR)|,|f'(qL)|)`，使精确膨胀波端点留在计算域内部。

精确压缩解为位于 `x=0` 的定常激波。精确膨胀解为稀疏扇：

```text
q(x,t) = qR                         if x/t <= f'(qR)
f'(q(x,t)) = x/t                    if f'(qR) < x/t < f'(qL)
q(x,t) = qL                         if x/t >= f'(qL)
```

稀疏扇内使用 `brentq` 反解 `f'(q)=x/t`；误差对照用高斯求积得到的精确 cell average，而非直接把中心点值当 cell average。

## 二次熵检查

取 `eta(q)=q^2/2`，满足 `Q'(q)=eta'(q)f'(q)` 的连续熵通量为

```text
beta(q) = 1 - (gamma - 1) M_inf^2 (q^2 - 1) / 2
Q(q) = q f(q) - beta(q)^(gamma/(gamma-1)) / (gamma M_inf^2)
```

将数值通量分裂为 `B(q)=-s(q)`、`A(q)=f(q)+s(q)`，使 `F(qL,qR)=B(qL)+A(qR)`。对应 E-O 熵通量按 `Q_B'=eta' B'`、`Q_A'=eta' A'` 积分，并用 `Q_A(q*)=Q(q*)` 固定常数。令

```text
H(q) = 0                                                     if q <= q*
H(q) = integral[q*,q] u m'(u) du                             if q > q*
     = q m(q) - q* m(q*) - [beta(q*)^r-beta(q)^r]/(gamma M_inf^2)
r = gamma/(gamma-1)
Q_B(q) = -H(q)
Q_A(q) = Q(q) + H(q)
G_EO(qL,qR) = Q_B(qL) + Q_A(qR)
```

每个 Euler 步、每个单元实际检查 fully-discrete 量

```text
D_i = eta(q_i^(n+1)) - eta(q_i^n)
      + (dt/dx) [G_EO(q_i^n,q_(i+1)^n)
                 - G_EO(q_(i-1)^n,q_i^n)]
```

记录所有步骤和单元的最大正违反量、违反计数及位置。阈值为 `max(max(D_i,0)) <= 2e-12`（本问题无量纲）。超过阈值即标 `FAIL`，保留数值和输出，不翻转熵通量符号、不删掉失败步，也不以 semidiscrete E-flux 测试代替该检查。若失败，须报告该熵通量不满足此 fully-discrete 检查，分析时间离散项；不得宣称 E-O 空间通量本身的 E-flux 判据失效。

## 验收阈值

- **预检：** `qR` 根残差与压缩 RH 速度绝对值均不超过 `1e-12`；`min f''(q)>0`。
- **稳定有界：** 所有时步状态保持 `[qR,qL]`，最大越界不超过 `5e-13`；代码不得做 clipping。
- **守恒：** 用累计边界通量修正的全域质量余额绝对值不超过 `5e-12`。
- **压缩激波：** 末态最大梯度面位置距 `x=0` 不超过 `dx/2`；处于端态 1%–99% 区间的单元宽度不超过 `2 dx`。
- **膨胀波：** sonic state `q*` 交点距 `x=0` 不超过 `2 dx`；报告扇宽及精确同定义宽度，L1/Linf 误差逐级网格记录。膨胀初值不得以反向尖锐跳跃作为接受解。
- **熵：** 每个单元/每步 fully-discrete 正违反量不超过 `2e-12`；否则熵验收失败，即使质量、界限和波形指标通过也不例外。
- **网格趋势：** 压缩位置/宽度和稀疏波 L1、Linf 误差都逐网格列出；稀疏波 L1 不得随加密增加超过 `1e-12`。不以此有限组网格替代渐近阶数研究。

程序与协议 SHA256、逐格网格解、时步历史、最终误差和所有验收状态写入 `potential/results/entropy_v1/`。整体状态只在全部指定算例通过时为 `PASS`；此状态仅代表本协议内的一维辅助演化。
