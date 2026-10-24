# 跨声速通量原始文献核查

## 结论

NASA TM-85751 给出了可复核的 Engquist-Osher (E-O) 全势通量偏置及弱解熵条件论证。它的对象是二维稳态、无旋、等熵全势方程的保守差分格式；文中还在广义坐标下写出同一通量构造。它不是可直接替换当前求解器面通量的一段坐标无关公式。当前 `potential/solve_airfoil.py` 在保角网格上以密度加权势差组装径向/周向共享通量，局部马赫数到声速时停止；E-O 要求依照方向和声速特征对质量通量作偏置，且证明依赖文中网格差分、度量项和弱激波假设。需要先推导与当前残差一致的二维度量通量，再验证离散熵不等式，不能仅把 `rho` 改成某个上风值。

TAWFIVE TM-4109 是另外一条有限体积实现参考：主单元内作三线性变量表示，在主单元面中点构造辅助单元并在其面上施加通量守恒，另加重耦合项以控制单点积分造成的奇偶点解耦；超声速区显式加人工黏性，并使用 FAS 多重网格。该内容支持其程序架构，不等于 E-O 通量，也不能证明上述辅助单元和 E-O 离散可直接拼接。

## Osher-Whitlow-Hafez：方程与 E-O 通量

以下符号沿原文。`q=|grad(phi)|` 是以自由来流速度归一化的局部速率，密度 `rho` 也按自由来流密度归一化。理想气体等熵关系给出

```text
rho(q) = [1 - (gamma - 1)/2 * M_inf^2 * (q^2 - 1)]^(1/(gamma - 1))
```

声速阈值（原文 `q* = a*`）为

```text
q* = sqrt([M_inf^(-2) + (gamma - 1)/2] / [(gamma + 1)/2])
rho* = rho(q*)
```

这来自原文方程 (2.2)、(2.5) 及印刷页 22 方程 (4.1)。对所讨论的正向一维速率，`q > q*` 等价于局部超声速 `M > 1`。

E-O 对乘积 `rho*q` 作声速分裂（方程 (4.2)）：

```text
(rho*q)_- = 0                         if q <= q*
(rho*q)_- = rho*q - rho* q*           if q > q*
```

印刷页 22 方程 (4.3) 的数值通量写成

```text
-h_EO(q_j, q_(j-1))
  = rho_j q_j - Delta_x * delta_x^- [(rho*q)_-]_j
```

其中 `delta_x^- f_j=(f_j-f_(j-1))/Delta_x` 是后向差分。令 `q1=q_j`、`q0=q_(j-1)`，并将 `(rho*q)_-` 记为 `s(q)`，逐项代入可得该标量两态通量：

```text
m(q) = rho(q) q
s(q) = 0                              if q <= q*
s(q) = m(q) - rho* q*                if q > q*
h_EO(q1,q0) = -m(q1) + s(q1) - s(q0)
```

所以第一个参数严格对应当前点 `j`，第二个参数对应后一点 `j-1`；原文 `h` 的符号是负质量通量，且一致性为 `h(q,q)=-m(q)`。原文称此通量单调；亚声速区退化为中心差分形式，超声速区偏置的是质量通量 `rho*q`，而不是单独偏置密度。公式符号由印刷页 22 图像核过；二进制文本抽取无法可靠保留上下标。

原文第 IV 节还比较 E-O 通量偏置与密度偏置。印刷页 28–29 给出的密度偏置写法使用局部马赫数表示，作者指出两种做法在亚声速中心差分下等价到二阶，但超声速区不同。E-O 与密度偏置的并列比较公式以未编号展示在印刷页 28–29；部分上标在扫描件中难辨，因此不把它们誊成更具体的离散公式或式号。原文明确强调 E-O 通量函数较光滑，利于迭代收敛；该性质不替代离散熵条件。

### Hafez-Osher (H-O) 式的证据边界

TM-85751 第 IV 节实际展开并给出式 (4.1)–(4.3) 的是 E-O flux；同文只在引言和参考文献 [15] 提到 Hafez、Whitlow、Osher 的平行论文 *Improved Finite Difference Schemes for Transonic Potential Calculations*, AIAA Paper 84-0092。NASA NTRS 记录 `19840036444` 的摘要称该论文采用基于 flux biasing 的 modified artificial density 方法，并报告其方案不产生膨胀激波，但该记录没有可下载全文。**因此 H-O 的精确偏置公式、方向定义、阈值及方程/页码在本次可核原文中均未知；不能将上面的 E-O 式冒称为 H-O 式。** 这是未取得该论文全文所致，不是对 E-O 扫描页 OCR 的猜测。

## 熵条件与使用假设

论文所称 entropy 是把全势问题改写为双曲守恒律系统后构造的凸数学熵，不是全势模型中的真实气体热力学熵增。原文 (2.9) 给出弱解熵不等式；(2.32) 将物理解激波条件写成该系统的熵判据。论文的结论是满足该判据可排除不物理膨胀激波，不表示无旋等熵全势模型能预测真实跨激波熵增。

论文将 Lipschitz 数值通量 `h(q1,q0)` 定义为 E-flux 的充分条件（印刷页 19，方程 (3.3)–(3.4)）：

```text
h(q,q) = -rho(q) q
sgn(q1 - q0) * [h(q1,q0) + rho(q) q] <= 0,
    for every q between q0 and q1
```

这里区间内的 `q` 是检验熵通量不等式的中间态，不是仅在两个端态各检查一次。若通量单调，即对第一参数非增、对第二参数非减，论文说明它是 E-flux 的特例。

从离散格式推出该熵性质时，论文定理 3.1 使用以下限制（印刷页 20–21）：

- 数值通量 `h` 对左右态 Lipschitz 且一致：`h(q,q) = -rho(q) q`，并满足原文定义的 E-flux 不等式；单调通量是其充分构造之一。
- 每个区域内离散势梯度各分量保持固定符号（假设 3.1）。
- 随网格缩小，局部离散振荡足够小，即文中所谓弱离散激波（假设 3.2）。
- 收敛定理还要求离散解及其离散梯度有界几乎处处收敛。

因此，此结论不覆盖任意强激波、任意振荡网格解、非一致通量或未经证明的多维通量组合。工程实现仍要检查质量守恒、正密度/正 Bernoulli 括号、熵残差和网格收敛。

## 二维保角网格对应关系

原文第 VI 节在印刷页 38 定义 `J`。其定义是**逆 Jacobian**

```text
J_inv = det[d(xi,eta)/d(x,y)]
U     = grad_x(xi)  dot grad_x(phi)
V     = grad_x(eta) dot grad_x(phi)
```

这里 `U,V` 是一般 contravariant 速度分量，不含 Jacobian 行列式。原文对应定义为 `U_i = sum_j a_ij * partial(phi)/partial(x_j)`、`a_ij = sum_k (partial(xi_i)/partial(x_k)) * (partial(xi_j)/partial(x_k))`，其中 `xi_i` 是计算坐标，`x_k` 是物理坐标。方程 (6.1) 的二维守恒形式因此是

```text
partial_xi (rho * U / J_inv)
+ partial_eta(rho * V / J_inv) = 0
```

若统一改用正向 Jacobian

```text
J_phys = det[d(x,y)/d(xi,eta)] = 1 / J_inv
```

则严格等价的形式为

```text
J_phys * div_x(rho * grad_x(phi))
  = partial_xi (J_phys * rho * U)
  + partial_eta(J_phys * rho * V) = 0
```

因此，`J_phys * div = partial(rho*U)` 漏了通量度量因子；使用 `J_phys` 时守恒通量必须是 `J_phys*rho*U`。只有采用论文的逆 Jacobian `J_inv` 时，才能将通量写成 `rho*U/J_inv`。论文 (6.2) 将 E-O 通量代入该变换后的方向差分；该式扫描字形不够清晰，本报告不硬抄其完整离散排版。

对项目的 Joukowski 坐标，令计算坐标为 `(r,theta)`，并明确采用正向面元 Jacobian `J_phys=det(d(x,y)/d(r,theta))`。则连续守恒通量分别为 `J_phys*rho*U_r` 和 `J_phys*rho*U_theta`，其中 `U_r=grad_x(r) dot grad_x(phi)`、`U_theta=grad_x(theta) dot grad_x(phi)`。这两个通量度量项不能遗漏。原文离散写法使用逆 Jacobian的除法形式，二者不可混记。

当前保角映射令计算网格为 `(r,theta)`，物理速度还要除以映射尺度 `|dz/dw|`。这意味着：

- 声速判断使用物理速度大小 `|grad_x(phi)|` 与 `q*` 比较，不能拿径向/角向分量各自和 `q*` 比。
- `r`、`theta` 面的守恒通量必须分别包含对应的 contravariant 速度分量与 `J_phys`（或严格等价的除以 `J_inv`），再与计算坐标面的有向差分配对；通量方向上的特征速度需由该面的法向物理速度及度量推导。
- `q*` 的一维 E-O 分裂不能未经推导直接施加到当前数组里的 `psi` 差分，也不能把两方向独立套用后就声称保留了二维定理。
- 只有当离散度量恒等式、面方向约定、面积/Jacobian 因子与单一共享面通量完全一致，并能重现论文证明所需的 E-flux 结构时，才可称为有文献依据的推广。论文提供坐标变换框架，但没有给出本项目特定 Joukowski 网格下的完整离散推导。

所以答案是：**可作为推导当前二维共享通量的理论模板，不可直接照抄为已经证明稳定的通量实现。** 现阶段应保留现有亚声速求解器边界；跨声速求解应在独立残差/测试分支验证后再接入。

## 最小一维压缩/膨胀激波核验

先在均匀笛卡尔一维网格实现论文 E-O 通量，不带保角映射、多重网格或二维边界。按同一 `M_inf, gamma` 由 Bernoulli 关系计算 `rho(q)`，每个单元使用唯一面通量并以相反符号累加到左右残差。至少完成以下判据：

1. **通量一致性：** 左右态相等时数值通量等于 `-rho(q)q`，并核验 `q <= q*` 的亚声速支和 `q > q*` 的超声速支在 `q*` 连续。
2. **压缩激波：** 取 `q_L > q* > q_R` 且满足一维稳态 Rankine-Hugoniot 质量通量相等的两态，以平滑/错误初值启动；收敛后检查远场两态、激波位置、每面通量一致和质量残差。论文印刷页 40 报告该类压缩激波用 Newton 法 39 次迭代到稳态，这是文献结果，不是本项目已复现实测。
3. **膨胀激波：** 把压缩解反向排列作为初值，而固定会产生物理解的边界态；迭代后不得保留稳定膨胀跳跃，应回到满足熵条件的稳态。论文印刷页 40 报告其示例 21 次迭代收敛。
4. **反向与网格检查：** 翻转网格编号和流向后应得到符号一致的镜像结果；逐次加密后比较激波位置、两态和守恒误差。报告离散熵不等式的离散求和值及正值违反量，不能只看残差或激波图像。

新增的 `potential/test_entropy_flux.py` 仅检查一维 E-O 标量数值通量的代数性质，不求解压缩/膨胀激波，也不验证二维坐标度量。该脚本对 `M_inf=0.5, 0.8`、`gamma=1.4` 的正 `q` 可容许域密集检查一致性、两参数单调性和 E-flux 区间不等式；通过只表示这些通量检查通过，不表示已解或验证跨声速流。压缩/膨胀激波演化、通量方向反转和网格加密仍是后续独立验证。

## TAWFIVE TM-4109 核对

手册印刷页 2（本地 PDF 第 4 页）明确记载其 conservative full-potential finite-volume 格式：单元内对自变量和因变量作三线性变化，用于计算 `rho,h,U,V,W`；辅助单元面位于主单元面的中间，并在辅助单元面上施加通量守恒；单点通量积分使奇偶编号点趋于解耦，因此添加保持二阶精度的 recoupling terms。超声速区域显式加人工黏性，形式模拟 Jameson rotated-difference 的迎风偏置；局部马赫数超过用户临界值后才启用，强激波时用户可选从二阶降为一阶黏性。

同页还说明其网格为绕翼型 C、翼展方向 H 的 C-H 拓扑，采用 SLOR 隐式线迭代，隐式方向可在周向/法向交替；非线性全势方程用 FAS 多重网格加速。该手册摘要/方法段足以核实这些具体做法，但不披露完整 FLO3OM 离散细节；它不是 E-O 熵证明的替代物。

## 本地证据与访问状态

- [Osher-Whitlow-Hafez NASA TM-85751 全文 PDF](../audit/potential_route/literature/19840012417.pdf)，58 页，1,242,870 bytes，SHA256 `B332EC4D370FDD2167F2E7116F4B984E7DFB2DD019322062CC0414E211C20436`。
- [Osher PDF 文本提取](../audit/potential_route/literature/19840012417.txt)：`pdftotext -layout` 本地提取，公式上下标需以 PDF 页面为准。
- [TAWFIVE NASA TM-4109 全文 PDF](../audit/potential_route/literature/19890011038.pdf)，67 页，2,577,519 bytes，SHA256 `2D191CE051A36D989CEF1992D97CF62DE137D52BBBEDD19DA96F6BAFB806076B`。
- [TAWFIVE PDF 文本提取](../audit/potential_route/literature/19890011038.txt)：`pdftotext -layout` 本地提取。
- 两个 PDF 均于本次任务通过用户指定代理 `http://192.168.3.9:7890` 从 NASA NTRS API 成功以 HTTP 200 下载；访问 URL、字节数和哈希见同目录 `access.json`。旧 METHODS.md 中的 403 是既往访问状态，不代表本次下载失败。
- H-O 并行论文的 NASA NTRS 记录为 `19840036444`（AIAA 84-0092）；记录未提供全文下载，因此未保存、未声称核对公式。

本文为中文技术概述和必要的短公式转写，不大段复述原文。页码均为文献印刷页，括号内 PDF 页码用于扫描件定位。

## 参考文献

1. S. Osher, W. Whitlow Jr., M. Hafez, *Entropy Condition Satisfying Approximations for the Full Potential Equations of Transonic Flow*, NASA TM-85751, 1984. 关键位置：方程 (2.9)、(2.32)、(3.1)–(3.6)、(4.1)–(4.3)、(6.1)–(6.2)，印刷页 8、17–22、38–40（PDF 页 9、18–23、39–41）。
2. N. Duane Melson, Craig L. Streett, *Updated Users' Guide for TAWFIVE With Multigrid*, NASA TM-4109, 1989. 关键位置：印刷页 2–3（PDF 页 4–5），全势有限体积、人工黏性、C-H 网格、SLOR 与 FAS 概述。
