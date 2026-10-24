# 独立 Prandtl 升力线参考协议

日期：2026-10-23；配对来源更正：2026-10-23。先冻结理论与算例协议，再创建独立脚本与结果。只使用 `lifting3d/lifting_line.py`、`LL_PROTOCOL.md`、`LL_RESULTS.md` 和带版本号的 `lifting3d/results/lifting_line_v*/`；不改 `lifting3d/solve.py`、网格生成器或既有三维结果。NumPy/BLAS 线程请求 NCPU=8。结果目录若已存在，脚本拒绝覆盖；修订配对证据写入新目录，不覆盖 `lifting_line_v1`。

## 模型与推导

采用稳态、不可压、无黏、小攻角 Prandtl lifting-line 模型。全翼展记作 `b`，半展向坐标为 `y∈[-b/2,b/2]`；弦长为 `c(y)`，来流速度 `U`。不使用、不拟合三维 solver 的输出作为模型输入。

有限翼用展向涡片表示。沿翼展的束缚环量 `Γ(y)` 在尾迹中产生 Biot–Savart 下洗；在翼面控制位置 `y0`，取主值积分

`w(y0)/U = (1/(4πU)) PV ∫[-b/2,b/2] Γ'(y)/(y0-y) dy`。

局部二维薄翼关系为 `Γ=0.5 U c a0 (α-αL0-αi)`，其中 `αi=w/U`。令

`y=(b/2) cos θ`, `0<θ<π`,
`Γ(θ)=2bU Σ[n=1..N] An sin(nθ)`。

将该有限正弦展开代入 Biot–Savart 主值积分，使用有限区间 Hilbert 核的正弦恒等式，得到

`αi(θ)=Σ[n=1..N] n An sin(nθ)/sinθ`。

代回局部二维关系并按正弦基展开：

`[4b/(a0 c(θ)) sin(nθ) + n sin(nθ)/sinθ] An = α-αL0`，

对每个 `θj=jπ/(N+1), j=1..N` 配点，按上式组成 N×N 线性系统。符号约定是正 α、正 Γ 给正升力，α 以弧度输入。对称翼型取 `αL0=0`。

由 Kutta–Joukowski 展向积分 `L=ρU∫Γdy`、参考面积 `S`，得

`CL=2∫Γdy/(U S)=π AR A1`, `AR=b²/S`。

由诱导阻力积分 `Di=ρ∫Γwdy` 及正弦正交性得

`CDi=2∫Γwdy/(U²S)=π AR Σ[n=1..N] n An²`。

脚本另用独立 Gauss–Legendre θ 积分重算升力和诱导阻力，以验证系数公式；不以代数残差代替气动适用范围。

## 预注册算例与门禁

| 顺序 | 算例 | 预设检查 |
|---|---|---|
| 1 | 椭圆弦长 `c(θ)=croot sinθ`，设 `AR=4`、`α=5°`、`αL0=0`；纯 `A1` 解析 patch | 解析 `A1=α/[1+4b/(a0 croot)]`，其余 `An=0`。在 N=16/32/64 检查离散矩阵对解析向量的最大方程残差、数值解 `A1`/高阶系数、模态公式 CL/CDi 和 θ 数值积分 CL/CDi；相对误差门限 `1e-12` |
| 2 | 矩形翼 AR=4、`c=1`、`b=4`，N=16/32/64，两个 `a0` 模型，α=`-5°,0°,5°` | 输出全部系数、Γ(y)、CL、CDi。N=32→64 的 `CL` 与 `CDi` 相对变化须 `<1e-3` 才称这两个积分量在本序列收敛；N=16 仅作趋势档。α=0 的系数绝对值及 ±5° 反对称误差另报，门限 `1e-12` |
| 3 | 同一 AR4 矩形翼，N=64；二维薄翼模型 `a0=2π/rad` 与 NACA0012 测量模型；α=`-5°,0°,5°` | 两模型分开报告，不择优、不从三维输出调参。NACA0012 输入取项目既有二维 case `airfoil2d/results/general_v1/20_n12_02_a5_m0/result.json` 的 M=0、α=5°、`CL=0.5990206726228917`，定义 `a0_NACA=CL/rad(5°)`，即约 `6.864/rad`。这是有限厚度二维 P2 的测量驱动近似，不当成普适精确 section slope |
| 4 | AR=8、20 矩形翼，N=64、α=`-5°,0°,5°`，两个 `a0` 模型 | 作为未来三维高展弦比对照与二维极限趋势记录；不设绝对真值门限 |
| 5 | 已保存三维 wing03、AR4、同批次同实现的 α=`-5°,0°,5°` | 逐站比较 Γ/(Uc) 及升力；报告 `CL_pressure`、`CL_jump` 和 `CD_pressure`，与升力线 CL、Γ、CDi 并列。配对证据必须同时核对 mesh、solver、P2、protocol SHA256，不可仅以同 mesh hash 推断同一实现版本。不同载荷定义、冻结尾迹及有限厚度可能造成差异；比较只作低阶一致性，不以 CDi 等同压力 CD |

经复核，三维载荷/环量比较与 ±攻角对称性只使用以下 `invariance_v1` 同批次结果。三者的 mesh SHA256 均为 `197df3425f6f13a53ac520689f6dbd1094e9fc25680e69f074e9c9bf3005d125`，solver SHA256 均为 `1d396d27dd23ef753af5043f40cd997517b65cb89b076a5e0c37b2e902ac0e66`，P2 SHA256 均为 `0047de5eb0a63768d97037e7a69b18defbc2a7da726044999e8aeba449da548f`，protocol SHA256 均为 `5d02e93691be6e2c3b4a6acf46cb186428bd06f5b24f179f90ddf6850a6307d5`；脚本对四项分别设相同性断言。相应快照 `solve.py.snapshot` / `p2.py.snapshot` 的文件 hash 与结果 JSON 中的实现 hash 一致。

- `lifting3d/results/invariance_v1/base/solve/result.json`（α=+5°）
- `lifting3d/results/invariance_v1/zero/solve/result.json`（α=0°）
- `lifting3d/results/invariance_v1/negative/solve/result.json`（α=−5°）

`initial/wing03_a5` 与 `paired_v1/wing03_a0`、`paired_v2/wing03_an5` 曾被用于 `lifting_line_v1` 展示，但不是同一实现版本，不再作为共同批次或对称门禁证据。`lifting_line_v1` 原始产物保留不覆盖；修订结果写入 `lifting_line_v2`。

另读 `invariance_v1/base/mesh.json` 确认 `span=4`、`chord=1`、`reference_area=4`。`paired_v1/wing015_curv32_a5` 的 α=5° 高分辨率结果只作旁列信息；其 metadata 表明同为 NACA 对称翼型、厚度 `0.12`、AR=4、弦长 1，区别是网格尺寸从 `0.30` 到 `0.15` 和曲率离散设置。不得描述成不同厚度或不同翼型比例。

## 解释限制与记录

本参考是独立、可追溯的线性升力线模型，不是三维全势/Euler 数值真解。两种 `a0` 的差异量化二维 section-slope 假设影响；NACA0012 测量值不是由三维结果拟合。有限翼尖诱导下洗包含在该理论中，但实际三维 solver 的有限厚度、固定尾迹、离散后缘条件和压力积分误差不包含在升力线模型内。

保存公式定义、参数与 α 单位、N/AR/a0 数据、全部 `An` 与展向 Γ、积分交叉核验、迭代/线性残差、三维源文件路径和 SHA256、比较数据、脚本/输入 hashes、NCPU 和耗时。报告中将解析 patch、截断序列收敛、三维 solver 对照分栏，不合并为一个“通过”数字。不得调整 `a0` 以靠近三维 solver 系数。

从项目根目录运行 `python lifting3d/lifting_line.py`。脚本只读取已保存的 wing03 结果和二维 NACA0012 JSON，不调用三维 solver 或网格 builder。
