# 不保存全矩阵的 GMRES、近场预条件与严格 XZ 半模

状态：可选研究后端；不改变生产 CLI、已验收 Torch 桥的默认求解、Cp 公式或物理精度状态。第一阶段仅追求相同离散方程上的数值一致性与更低内存，不承诺比小规模 LU 快。

## 已有本地基线与本次范围

GRASS PR26 `e4add49` 已有真实导入验收：10 passed，0 skipped；全仓839 passed、2既有skip。完成的是首翼段内交线**采样环与一阶几何VJP**，还没有机身开孔、外露翼面/机身共享三角网格、无埋入面审计及完整公共外表面气动接线。

Dpanel 本地 `d3a1d6` 的目标块+checkpoint已通过对应回归：1976壁面，VJP峰值从4174.7 MB降至block32的584.4 MB、block64的612.0 MB（各约11.8s墙钟）。这些是该次CPU进程测量，不是本PR的新结果。稠密A和LU仍保留；所以不能把内核临时量下降表述为整个算法从O(N²)变成O(N)。

本PR堆叠于PR15 `49f5b36`，纳入上述已验证分块核，并只扩展全局面片ID以正确处理非连续/矩形子块的内部自影响。新增算子路径调用同一解析核和同一压力恢复，不复制物理公式。`panel/solve.py`不改，`panel/differentiable.py`把原有压力后处理抽成共享`recover_loads`，稠密路径输出契约不变。

## 方程和存储

原系统为 A=[[D,W],[-T,M]]，右端为[B sigma,0]。新`PanelOperator`只提供`rhs()`、`matvec()`、`rmatvec()`，每个目标块当场生成解析系数并与向量收缩，立即丢弃该块；不拼接全域B、D、W或全局LU。尾迹线性环量通过站位插值/散射实现，Kutta通过边积分/散射实现，不保存T的稠密大矩阵。

若目标块大小b、预条件近场块大小p、重启维数m、环量未知数G，主要设计存储约为O(b(N+Nwake)+Np+Nm+G²)，另有几何和局部后处理。原有hat_map为稠密站位×G，环量质量块仍为G²；G显著增加时它们也需改进。实际XLA/AD峰值须测，不能仅据此公式宣称实测内存下降。

每次作用仍需全部成对影响，工作量O(N²)，不是FMM。大系统可能以内存换时间；FMM/层级低秩与更强预条件另立题，不在第一版偷偷近似远场。

## 预条件

冻结基准几何上的空间分区，使同一近场块尽量包含相邻上下表面。预条件矩阵为P=[[B_local,0],[-T,M]]：壁面近场子块各做小LU，环量质量块做小LU，保留Kutta下三角耦合。没有对角加epsilon、SVD裁剪或奇异块fallback；失败需报告。

分别提供P⁻¹和P⁻ᵀ，不是将同一个左预条件器套在正向和非对称伴随问题上。LU因子停止梯度只用于数值求解路径；真正导数来自原始A(X)、b(X)的隐式方程，不是预条件近似的导数。正确性用小型显式P和Pᵀ对照验证。

## 严格半模

半模算子是A_half=R A E：E把同一未知量分配给两个镜像物面/环量位置，R选择独立配点与Kutta方程。转置必须为Eᵀ Aᵀ Rᵀ，即把两侧的余切相加；不能只取右侧梯度。中心线环量自由度可能自映射，所以总未知量不一定恰好除以二。新测试的40壁面+3环量=43未知量降为20壁面+2环量=22。

初版保留完整左右几何O(N)，但未知量、配点行和Krylov向量减少；每个半模目标块仍需计入两侧源贡献。因此**不声称内存再降4倍或总耗时降8倍**。这些比例仅属于完整稠密矩阵与LU理论，不适用于这里的分块矩阵无关路径。

必须是几何、三角剖分、外法向、TE上下邻接、尾迹和环量基函数均严格镜像，且零侧滑/对称工况。不对称网格或不同镜像三角对角线一律拒绝；不通过平均AIC或放宽容差伪造等价。不在y=0加虚假壁面。只允许保持镜像关系的几何设计方向；一般非对称设计导数必须使用全模。

## 自动微分与JAX兼容

复用公开`jax.scipy.sparse.linalg.gmres`和`jax.lax.custom_linear_solve`；后者要求数值正向/转置解真正满足方程。两次求解都直接检查原方程残差，未达阈值返回NaN，验收驱动拒绝结果，不回退LU冒充成功。

JAX0.9.0.1在自动转置部分带闭包的分块循环时触发`ValAccum`/scan错误。因此原生路径的`linear_action`显式登记已验证的分块转置，并用custom JVP保留几何导数；不是将非对称系统标为symmetric=True。

**2026-10-25 JAX 0.10.2 兼容修复**：本地实测该版本不存在公开的`jax.custom_transpose`模块，原PR在pytest collection就报`ModuleNotFoundError`。现有`panel/iterative.py`采用能力检测：支持`jax.custom_transpose`的版本仍使用原生JAX GMRES/custom_linear_solve；没有该公开接口的版本在**CPU**上改由`jax.pure_callback`运行SciPy GMRES，调用同一流式JAX `Av/Aᵀv`与`P⁻¹/P⁻ᵀ`，并由`jax.custom_vjp`完成一阶隐式伴随。伴随对`A(X)u`做真正的几何VJP，rhs几何导数从求解函数的独立输入自动回传；**拓扑索引固定、不求导**。该兼容路径不形成稠密AIC、不对矩阵设对称、不把预条件器当真实算子，但包含Python主机回调/向量拷贝、额外编译和重算，不能等价宣称JAX原生速度或GPU能力。

若SciPy GMRES未收敛或真残差超出阈值，兼容路径返回NaN供外层拒绝；不自动切换LU、不放宽容差。不支持正向模式JVP和高阶导数。新增`test/test_iterative_jax010_compat.py`可以在有/没有公开custom_transpose的JAX环境中强制走CPU兼容路径，验证全模、半模状态/转置伴随及几何VJP。**提交不等于本地JAX0.10.2已跑通；必须取得本地真实回执。**

JAX GMRES的info不是可移植的实际迭代计数。驱动的`--count`额外调用SciPy GMRES，使用完全相同的JIT解析算子/预条件，以回调记录迭代和matvec；它是**另一条计数诊断**，不能把其迭代数写成JAX内部计数，也不能混入JAX核心时延。

## 本次云端小型测试（不是本地真实翼身验收）

Python3.13.5 / JAX0.9.0.1 / NumPy2.3.5 / SciPy1.17.0 / CPU Float64：`python -m pytest -q test/test_iterative_panel.py` 实跑 **10 passed / 178.63s**。夹具40壁面+8尾迹，完整状态43、严格半模22。覆盖矩形块自影响、Av/Aᵀv及点积、P⁻¹/P⁻ᵀ、全模/半模状态与原始载荷、转置伴随、合法联合几何VJP对LU及三档FD、失败拒绝。另有JAXPR数组形状检查，**不是**完整XLA/GPU峰值内存证据。

单独核对抽取`recover_loads`前后的旧稠密函数：该夹具Cp/面力/面矩/状态/矩阵/右端最大差均为0。相同单位来流下，全模无预条件、全模有预条件、半模有预条件的真相对残差分别2.84e-12、1.22e-12、1.06e-12，最大Cp差约1.41e-12、9.79e-13、1.02e-12。这里仅说明同一离散方程数值一致，不代表真实气动物理准确。

## 本地运行

先在独立工作树、现有JAX CPU Float64环境（无需CAD/新依赖）：

```bash
python -m pytest -q test/test_iterative_jax010_compat.py test/test_iterative_panel.py test/test_differentiable_panel_fields.py
```

实际已有单翼484面、双机身616面、前置机身696面，分别在新进程/新输出目录运行：

```bash
# 新分块kernel的LU参考；不得拿旧未分块高峰值作不公平基线
python -m panel.verify_iterative --run <saved solve> --out runs/new_lu \
  --mode lu --pc none --block 32 --near 32 --restart 40 --cycles 20 --tol 1e-10 --derivatives

python -m panel.verify_iterative --run <same saved solve> --out runs/new_mf_none \
  --mode gmres --pc none --block 32 --near 32 --restart 40 --cycles 20 --tol 1e-10 --derivatives

python -m panel.verify_iterative --run <same saved solve> --out runs/new_mf_block \
  --mode gmres --pc block --block 32 --near 32 --restart 40 --cycles 20 --tol 1e-10 --derivatives

# 只对满足严格镜像离散的同一个几何；不匹配即blocked，不重新剖分冒充同离散对照
python -m panel.verify_iterative --run <same saved solve> --out runs/new_mf_half \
  --mode gmres --pc block --block 32 --near 32 --restart 40 --cycles 20 --tol 1e-10 --half --derivatives
```

需要迭代次数时另加`--count`进行独立计数。先从小算例验证收敛再扩大规模，禁止直接跑23536面。半模被拒不应阻塞全模矩阵无关测试；不允许静默改为全模后仍标half。

报告HEAD、依赖、BLAS线程、mode/块/restart/容差、正向和转置真残差、Cp/面力矩/状态对原场的误差、对称合法方向VJP与三档FD、冷/热前反向及父进程峰值RSS。进程中不要先调用完整稠密参考再测迭代峰值，否则高水位不可比较。

## 仍未交付

真实大翼身的内存/性能收益、所有现有非镜像网格的半模化、GPU、FMM、自由尾迹、自动多翼拓扑、生产Torch桥接默认切换、气动精度与正式训练。物理精度标记始终false；这次只解同一个低阶离散问题。

## 主要参考

- JAX GMRES（可调用算子、预条件、重启与info边界）：https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.gmres.html
- JAX custom_linear_solve（原方程与转置伴随不变量）：https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html
- JAX custom transpose设计：https://github.com/jax-ml/jax/issues/9129
- SciPy GMRES（真实残差与callback_type）：https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html
- 本地内存基线：Dpanel `d3a1d6`，docs/KERNEL_MEMORY_BLOCKING.md。
