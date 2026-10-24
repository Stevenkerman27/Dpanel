# Airfoil2D 修复版实验协议

日期：2026-10-23。先记录协议，再修改 `solve.py`、`build_mesh.py` 并执行实验。旧 `meshes/extra_v1/`、`results/extra_v1/` 只读保留；首轮实现证据 `extra_repair_v1` 也保留。本协议对应的最终修正版网格和结果写入 `meshes/extra_repair_v2/`、`results/extra_repair_v2/`。线性脚本请求 NCPU=8。除特意测试 NCPU 外不改变线程设置。

## 假设与实现边界

1. 环量中心必须在做 winding/角度运算前逐坐标检查有限性。NaN/Inf 与有限但体外中心使用不同失败诊断和非零退出。
2. Kutta 离散条件沿几何弦方向求迹差。方向由实际轮廓的 leading-edge 点指向 trailing-edge 点，归一化后用于 `te_gradient @ chord_direction`。旋转网格与来流同步旋转后，均匀边界解应在舍入误差内保持系数和旋回后的 Cp；旋转网格的解析 exact 边界元数据不受支持，必须拒绝而非沿用未旋转映射。
3. 网格远场尺寸与外半径分离。新增 `--far-size`，默认 `1 chord`；Gmsh `MeshSizeMax` 和阈值场 `SizeMax` 均由该值控制，不得随 R 变化。近场阈值保持 `SizeMin=size*chord`、`DistMin=.05 chord`、`DistMax=2 chord`。
4. 不改 circulation center、不通过 Cp 裁剪或数值稳定参数掩盖物理解差异。此轮不实现中心 gauge 变换；中心差异按 P2 网格加密序列测量。若将来实现 gauge，须另立协议，证明 `g_cd(z)=arg((z-c)/(z-d))` 为域内单值函数且 `grad(g_cd)=h_c-h_d`，并验证解析梯度与解析流场，不能用其 P2 插值梯度替代。

## 预先固定的实验

除另有说明，合法算例使用 `M=0`、`alpha=5 deg`、quadrature=4、NCPU=8。

| 组 | 固定输入 | 预先门禁/报告方式 |
|---|---|---|
| NaN/Inf 中心 | R8 网格，分别 `--center nan 0`、`--center inf 0`；另保留有限体外中心对照 | 非有限输入须在 winding 前以专门 finite 诊断拒绝；有限体外输入仍报告体内 winding 失败。均须非零退出、不得产生收敛结果 |
| Kutta 旋转 | 新建 R8 对称 Joukowski `epsilon=.08, eta=0, size=.02, far-size=1` 网格；基准 uniform alpha=5°；旋转 20° 和 45° 的网格/轮廓并同步旋转来流角 | 比较 `CL_pressure`、`CL_circulation`、`CD_pressure` 差 `<.002`，旋回坐标的 wall Cp 最大差 `<.03`。uniform 分支检验旋转协变；旋转后 exact 解析映射必须明确拒绝 |
| 远场及网格尺度 | 对称 Joukowski `.08/0`；R=`4,8,16 chord`；每个 R 均生成一张 `size=.02 chord, far-size=1 chord` 网格，并在同一网格分别跑 uniform/exact | 保存网格 SHA256、近场边长统计、P2 单元数、最大物理边长及实际 MeshSizeMax/阈值参数。每种边界条件分别要求 R8/R16 `abs(delta CL_pressure)<.002`；R4 只报告，不要求误差单调。每个工况都与同网格 exact 参考比较 CL、CD、Cp RMSE；不以收敛状态代替物理对照 |
| 环量中心网格趋势 | 对称 Joukowski `.08/0`、R8、far-size=1；h=`.02,.01,.005 chord` 各建一张网格，每张网格以 exact 边界分别使用默认中心、`(.2,0)`、`(-.5,0)` | 每档相对同网格默认中心报告压力/环量 CL、CD、Cp 最大差与 RMSE。绘列或表列差异随 h 的变化；不得用拟合曲线消去差异，不得调整门限或强制中心等价。解释其是否随离散加密缩小；若不呈收敛趋势则登记为未解决数值敏感性 |

网格构建必须记录 `far_size_chord=1`，并验证 `MeshSizeMax=far-size*chord`、Threshold `SizeMax=far-size*chord` 与 R 无关，`DistMin/DistMax` 分别固定为 `.05/2 chord`。同 R 下 exact/uniform 必须共用完全相同 mesh SHA256。跨 R 网格不要求 SHA256 相同。

## 证据与判断

修复前 `extra_v1` 中 R16 只有 320 单元，而 R8 有 1080 单元；本轮的核心因果对照是固定 far-size 后重复半径序列。若 R16 仍不满足旧 `.002` 门禁，报告失败，不放宽阈值。该门禁只评价本研究网格与边界半径稳定性，不是气动绝对真值认证。

所有求解输出保存命令、return code、elapsed time、mesh/source/protocol SHA256、完整 history、系数、Cp finite/extrema/RMSE、壁面误差、弱式与 raw-flux 指标、日志以及失败 JSON。预期拒绝与合法收敛分开计数。summary 不得把非零退出自动视为通过，也不得把零退出或残差合格称作物理正确。

运行脚本前目标 `extra_repair_v2` 目录必须不存在，以保留已生成证据；从项目根目录运行 `python airfoil2d/run_extra_repair.py`。本协议及门禁冻结后才开始代码编辑和构网。
