# 一般二维鲁棒性追加协议

日期：2026-10-23。此协议先于本阶段新增网格与求解运行。所有新文件写入 `airfoil2d/meshes/extra_v1/` 与 `airfoil2d/results/extra_v1/`；不修改 `solve.py`、`build_mesh.py`、既有结果或根目录文档。运行线程数固定请求 NCPU=8；另设单线程对照时只将该对照设为 1。

## 问题与已知边界

当前离散 Kutta 行在后缘节点计算相邻单元的 `te_gradient[:, 0]`，即全局 x 方向导数差。对于弦线平行 x 轴的翼型，这与弦向切向相符；对旋转翼型，物理弦向不再是全局 x。构造旋转测试时只旋转已保存的 P2 网格坐标及轮廓，不经 `build_mesh.py` 重建、不改边/节点拓扑；对几何和自由来流同时旋转相同角度，再比较载荷与旋回坐标后的 Cp。若超过下列阈值，记为 Kutta 方向导致的旋转协变性失败；不得改 Kutta 行、重建或筛掉失败后再报告。

Joukowski 的 `eta != 0` 解析分支由现有 `solve.py` 支持，解析速度、解析环量与精确远场边界都直接使用同一个复映射参数 `c=-epsilon+i*eta`。本次用 `eta=0.04` 检查弯翼映射、解析外边界与弦向力系数。`epsilon=0.30` 用于较厚但仍锐后缘的 Joukowski 翼型。生成网格仍是样条近似，故同时保留解析误差与网格元数据。

## 固定算例

所有成功求解的基准参数为 `M=0, alpha=5 deg, quadrature=4, maxiter=100`，除非表内说明。精确参考仅用于解析 Joukowski 两例，不将解析边界选项用于 NACA 或旋转几何。

| 项目 | 输入/变换 | 预设评价 |
|---|---|---|
| 弯翼解析参考 | Joukowski `epsilon=.08, eta=.04`, 网格尺寸 `.02 chord`, `R=8 chord`, exact 边界 | solver 收敛；环量 CL 与压力 CL 各自相对解析 CL 误差 `<1%`；Cp 加权 RMSE `<.03`；`abs(CD_pressure)<.005`；记录 Kutta、弱式残差、网格 Jacobian |
| 厚翼解析参考 | Joukowski `epsilon=.30, eta=0`, 同网格尺寸/R/exact | 同上。该门禁评估指定解析几何及样条网格，不代表一般厚翼认证 |
| 远场半径 | 对称 Joukowski `.08/0`, uniform 边界；R=`4,8,16 chord`，相同尺寸比 `.02` | 三例均应收敛且有限；报告 CL、CD、Cp/壁面误差及网格规模。R8 与 R16 的 `abs(delta CL)<.002` 为远场稳定性门禁；R4 单独报告，不要求误差单调 |
| NCPU 一致性 | 复用弯翼同一 mesh 文件及同一 exact 工况，分别 `--ncpu 1/8` | 两次均收敛；CL、环量、CD 最大绝对差及保存 Cp 最大差各 `<=1e-10`。输入网格 SHA256 必须相同 |
| 环量基中心敏感性 | 复用 R8 对称 Joukowski 网格和 exact 工况：默认中心、`(0.2,0)`、`(-0.5,0)` | 三点必须由 winding 检查认定在体内并收敛；相对默认中心的 `abs(delta CL_pressure)<.005`、壁面 Cp 最大差 `<.05`。同时记录各自 Cp RMSE、环量、Kutta 和弱式残差；这只检查当前离散对基函数位置的敏感性 |
| 几何旋转协变性 | 复用 general_v1 的 Joukowski j02 固定网格和 uniform 边界；旋转 `20 deg`，同时令 alpha 从 `5 deg` 变为 `25 deg`，中心和点坐标一并旋转 | 比较 CL pressure、CL circulation、CD 的绝对差均 `<.002`，旋回坐标后 Cp 最大差 `<.03` 才记通过。预期判据不因已知全局 x Kutta 行而放宽；不通过就是本版本的明确方向性限制 |

坐标旋转绕 mesh `normalization_origin` 进行。变换后将 metadata `shape` 置为 `rotated_file` 并使用 uniform 边界，阻止现有解析后处理错误地套用未旋转的解析映射；拓扑数组原样复制。

## 非法输入分类

每个输入独立进程、独立输出目录，保留 return code、stdout/stderr 和 `failure.json`（若程序写出）。以下均应以非零状态拒绝，且不得生成 `converged=true`：

| 标签 | 输入 | 预期失败类别 |
|---|---|---|
| `invalid_mach` | `--mach 1` | 参数范围断言 |
| `invalid_quadrature` | `--quadrature 2` | 参数范围断言 |
| `outside_center` | `--center 10 10` | `Circulation center must lie inside the body` |
| `nonfinite_center` | `--center nan 0` | 应拒绝非有限中心；单独记录实际消息，若被误报为体外中心则登记分类缺陷 |
| `unsupported_exact_geometry` | NACA mesh 加 `--boundary exact` | 解析参考适用范围断言 |

预期拒绝按测试通过记录，但与合法算例收敛分开统计。若进程崩溃、写出非有限 `result.json`，或退出 0 却无收敛/失败记录，列为缺陷，不重试到“成功”。

## 运行与原始证据

`run_extra.py` 先生成本阶段专属网格，再顺序调用现有 builder/solver；builder 与 solver 源文件只读。每个 mesh 和 case 保存命令、返回码、运行时间及输入/代码哈希；case 保存 solver 原始 JSON、日志与场数据。汇总门禁在结果写完后从原始结果计算，预设阈值不按输出后调。

不重复 general_v1 的 74 条混合检查，也不把此前 10 条变换检查重跑。本阶段的实验成功与否分别报告；物理解正确性只能由弯翼/厚翼解析比较支撑到列明范围，不由退出码、残差或线程一致性单独认定。
