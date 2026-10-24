# Airfoil2D 修复版追加结果

日期：2026-10-23。先冻结 [REPAIR_PROTOCOL.md](REPAIR_PROTOCOL.md)，再修改 solver/builder。旧 `extra_v1` 原始结果未改；首轮修复证据 `extra_repair_v1` 也保留，但因发现 LE metadata 重复缩放而不作为最终验收。本报告只采用最终 `extra_repair_v2`。

## 计数与范围

`summary.json` 的 **28/28 是运行和预设对照门禁记录，不是 28 个气动算例**。其中有 22 个 solver 进程：18 个合法算例均收敛且 wall Cp 有限；4 个预期拒绝分别为 NaN 中心、Inf 中心、有限体外中心、旋转几何的 exact 边界。其余 6 项是远场半径对照、Kutta 投影旋转恒等式和旋转协变性检查。所有合法算例请求 NCPU=8。

退出码、弱式残差和 `converged_subsonic` 只评价执行/代数状态，不单独证明物理解正确。summary 和每个 case 的 `result.json` 保留完整载荷历史、残差、Cp 场及输入/源文件哈希。

## 修复

`solve.py` 在 winding 检查之前验证环量中心坐标有限。NaN、Inf 均以非零码退出并给出 `Circulation center coordinates must be finite`；有限体外中心仍单独报告 `Circulation center must lie inside the body`。旋转几何的 exact 映射分支以清楚诊断拒绝。

Kutta 行现在将后缘迹梯度差投影到单位弦向。`build_mesh.py` 保存 contour 上实际的 leading-edge 坐标；solver 从该点指向 trailing-edge 节点。检查了各 repair 网格的 metadata LE 与 contour 重合，距离为 0。首轮 repair 的 metadata 曾重复乘 chord；该批原始结果保留但不用于本报告。

## 旋转协变性

同一 R8 固定网格旋转几何 20°/45°，来流角同步从 5° 调整至 25°/50°，采用 uniform 边界：

| 旋转角 | `abs(delta CL_pressure)` | `abs(delta CL_circulation)` | `abs(delta CD_pressure)` | 旋回坐标 wall Cp 最大差 |
|---:|---:|---:|---:|---:|
| 20° | `6.22e-15` | `9.77e-15` | `3.19e-15` | `1.80e-12` |
| 45° | `4.13e-14` | `6.47e-14` | `1.72e-15` | `2.33e-12` |

两角都远低于原门限 `.002/.03`；单独的向量投影旋转恒等式误差为 `2.22e-16`。这验证了当前固定网格、uniform 边界和所测姿态的协变性，不扩展为任意网格/几何认证。旋转网格 exact 边界按协议拒绝。

## 远场半径与网格尺度

新增 `--far-size` 默认 `1 chord`。本组各 R 的 `MeshSizeMax` 和阈值 `SizeMax` 均为 `1 chord`；`DistMin=.05 chord`、`DistMax=2 chord` 不随 R 变。三张 `.02 chord` 网格近壁边最大长度均为 `.019885 chord`。单元数随外域规模增加：R4 1752、R8 2114、R16 3524；最大顶点边长分别为 `1.150、1.330、1.339 chord`。这与旧版 R16 仅 320 单元形成对照，表明旧设置的远场粗化耦合已解除到本网格尺寸场范围内。

| R / chord | 边界 | 单元数 | 压力 CL | 环量 CL | 解析 CL | 压力 CD | Cp RMSE 对解析解 | 壁面法向最大值 / U∞ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | uniform | 1752 | 0.580684 | 0.581121 | 0.588180 | 0.000618 | 0.014459 | 0.14129 |
| 4 | exact | 1752 | 0.586252 | 0.586279 | 0.588180 | 0.000200 | 0.012628 | 0.14202 |
| 8 | uniform | 2114 | 0.584825 | 0.584985 | 0.588180 | 0.000336 | 0.013041 | 0.14245 |
| 8 | exact | 2114 | 0.586234 | 0.586290 | 0.588180 | 0.000232 | 0.012782 | 0.14264 |
| 16 | uniform | 3524 | 0.585861 | 0.585927 | 0.588180 | 0.000183 | 0.012493 | 0.13378 |
| 16 | exact | 3524 | 0.586214 | 0.586255 | 0.588180 | 0.000157 | 0.012449 | 0.13383 |

均为同一 R 网格上分别运行 uniform 和 exact，因此每对输入 mesh SHA256 完全相同。uniform 相对 exact 的压力 CL 差从 R4 `.005567`、R8 `.001409` 降至 R16 `.000353`；wall Cp 最大差从 `.03937`、`.00984` 降至 `.00253`。R8→R16 压力 CL 差在 uniform/exact 下分别为 `.001036` 和 `.0000197`，都通过原有 `<.002` 门限。该结果支持固定远场尺寸后边界半径影响减小；有限半径序列仍不是连续外域极限或更广泛气动真值认证。

## 环量中心与网格趋势

无 gauge 变换。连续问题中令 `h_c=grad(arg(z-c))`，两个均位于实体内部的中心 `c,d` 满足：

`g_cd(z)=arg((z-c)/(z-d))`，`grad(g_cd)=h_c-h_d`。

因此 `grad(psi_c)+k h_c = grad(psi_c+k g_cd)+k h_d`：中心变化可由单值势函数 gauge 精确吸收。离散 P2 势不能精确表示一般的 `g_cd`，故有限网格解仍会对中心有差异；这不是将中心强制设为默认值的理由。下表是在每档同一网格上相对默认中心的测量，无绝对中心不敏感门限：

| h / chord | 中心 | 压力 CL 差 | 环量 CL 差 | wall Cp 最大差 | 默认/偏移中心 Cp RMSE |
|---:|---|---:|---:|---:|---:|
| .020 | `(.2,0)` | `1.315e-5` | `1.009e-5` | 0.13712 | 0.01278 / 0.01536 |
| .020 | `(-.5,0)` | `2.187e-5` | `1.670e-5` | 0.08677 | 0.01278 / 0.01050 |
| .010 | `(.2,0)` | `1.573e-5` | `1.359e-5` | 0.04023 | 0.00466 / 0.00514 |
| .010 | `(-.5,0)` | `2.418e-5` | `2.018e-5` | 0.02510 | 0.00466 / 0.00423 |
| .005 | `(.2,0)` | `1.055e-5` | `0.921e-5` | 0.01077 | 0.00243 / 0.00250 |
| .005 | `(-.5,0)` | `1.912e-5` | `1.594e-5` | 0.00670 | 0.00243 / 0.00237 |

Cp 最大差在每次 h 减半时约缩小 3.4 倍，三个采样档位呈一致下降；CL 差很小但不是严格单调。这里不作拟合或外推；证据与 P2 对解析 gauge 势逼近误差随网格加密降低相符。尚未用独立高阶离散验证收敛阶，也未实现解析 gauge-enriched trial space。

## 拒绝分类、证据与复现

NaN/Inf 中心、体外中心、旋转 exact 分支各自非零退出且失败消息分类正确，无收敛结果。NACA/旋转映射分支仍不在本次 exact 解析边界适用范围。

最终证据位于 `meshes/extra_repair_v2/` 和 `results/extra_repair_v2/`。`summary.json` 记录命令、return code、耗时、history、CP finite/extrema、网格统计、完整输入和源哈希；case 目录保留 stdout/stderr、result/failure JSON 与 field NPZ。运行器 SHA256：`630eb9ae72f213a53cf279e7062d8ce8718c12426a80742e4f8da30896ffa469`；solver：`edd2158669d5c118081c991b837b5579679e41101cbe554421de3ae48f308989`；builder：`d1dc30eacc1b317dce1f685f3d0ebbcb0a5378d731bc2ea032739cee942ad52f`；协议：`abc4cb200a56c8385a09050fb11cd89049794368c04803ae01d31eb7fa7d6033`。Python executable 为 `C:\Users\zyx20\anaconda3\python.exe`，SHA256 `7679e53fa969789309e81fdad0d52b8cda5f83c9abf7cb31a3c58bf24b31e264`。

在干净工作副本中，从项目根目录运行：

```powershell
python airfoil2d/run_extra_repair.py
```

脚本保护已有 v2 证据，目标目录已存在时会拒绝运行。

## 未完成

中心结果只覆盖指定对称 Joukowski 几何、三个 h 和两个偏移中心；没有 gauge-enriched P2 实现、解析 gauge 边界项的独立测试或更高阶误差认证。远场研究限于 R4/8/16、固定 size/far-size 的单一网格族。旋转测试限于 20°/45° uniform 分支；exact 旋转分支明确拒绝。粘性、跨声速/激波、复杂多元件和三维问题均未覆盖。
