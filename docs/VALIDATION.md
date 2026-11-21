# 测试与验证

现役验证的参照是无粘解析解、升力线和 VSPAERO。风洞数据作为历史资料归档。

| 层级 | 检查对象 | 判据 |
|---|---|---|
| 微观 | 核函数、矩阵装配、约束、Kutta 闭合、尾迹推进、反传和接口 | 精确积分或制造解、对称性与守恒等不变量、零空间约束、独立中心差分、错误输入与失败状态 |
| 宏观 | 从几何到压力、升力、环量和诱导阻力的整条求解链 | 解析解及网格收敛、升力线的适用范围与误差带、VSPAERO 同工况对照 |

测试按断言对象分层。三角环量的解析 Trefftz 积分属于微观测试，整翼的展向环量和诱导阻力属于宏观测试。小网格上的固定与自由尾迹初态一致性、前向与伴随接口一致性也属于微观测试。耗时和显存由独立 benchmark 记录。

解析标模记录误差及其随网格细化的变化。升力线检验小迎角、有限翼展下的升力和环量趋势，容差应同时考虑模型近似与网格误差。VSPAERO 对照应记录版本、几何、工况、尾迹设置、参考面积与力矩参考点，并采用一致的载荷定义。先对齐升力斜率和展向环量，再评估压力和诱导阻力的可比性。各类判据使用明确的绝对或相对误差，来源是理论、离散阶次及网格研究。

## 自动化测试

`macro` 选取端到端气动比较和网格细化用例。默认主线覆盖球体的解析压力、薄翼的升力线斜率和细化趋势、椭圆翼的环量与诱导阻力，以及自由尾迹在零迎角的对称性和尾迹平衡。配点路径的两项严格预期失败继续由原用例跟踪。尖翼梢 Jacobian 条件和残差检查属于微观测试。

`tools/calibration.py` 复用球体、薄翼和椭圆翼的求解与测量，并输出完整标模报告。`tools/acceptance.py` 的内部选项对照用于诊断。标模判据与网格记录见 [CALIBRATION.md](reference/CALIBRATION.md)。执行状态见 [STATUS.md](../STATUS.md)。

二维圆柱和 Joukowski 对照分别见 [全势流圆柱结果](../potential/RESULTS.md)和
[P2 翼型结果](../airfoil2d/RESULTS.md)。

## VSPAERO 对照

`tools.vspaero_compare` 从现有案例清单读取几何与参考量。DPanel 使用固定尾迹 `Galerkin + pressure + area`，VSPAERO 使用两档厚面元网格和一次尾迹迭代。默认对照矩形翼和椭圆翼。还可以显式选取 `swept30` 和 `bwb` 用于诊断，这两组的 VSPAERO 重建省略了翼型弯度。

报告记录升力曲线斜率和 4° 的归一化展向环量误差。DPanel 采用压力积分 \(C_L\)，VSPAERO 采用 `CLwtot`，环量由 `.lod` 的分段升力与弦长重构。尾迹离散和升力计算口径的差异也会影响结果。早期结果见 [VSPAERO 对照记录](archive/issue/ISSUE92_B3_VSPAERO_PROTOCOL.md)。

OpenVSP 3.56.0 实测使用 0°、2°、4°、6°。下表使用 VSPAERO 细网格，斜率差以 VSPAERO 为分母。

| 构型 | VSPAERO 面数 | 升力斜率差 | 4° 环量 RMS 差 |
|---|---:|---:|---:|
| 矩形翼 | 1360 | −5.41% | 0.0473 |
| 椭圆翼 | 6080 | +0.15% | 0.0402 |
| 30° 后掠翼 | 2480 | −1.27% | 0.0146 |
| 翼身融合体 | 7280 | −10.15% | 0.4089 |

翼身融合体在粗、细两档的环量 RMS 差分别为 0.388 和 0.409。该构型需要先对齐翼型弯度和条带口径，再评估分布差异。

## 复现

```powershell
python -m pytest -q -p no:cacheprovider --strict-markers -m "not macro" test
python -m pytest -q -p no:cacheprovider --strict-markers -m macro test
python -m tools.calibration --out runs/verification/macro_calibration
python -m tools.calibration --case C05 --refine --out runs/verification/macro_calibration_refine
python -m tools.vspaero_compare --vsp-root D:/3D/Projects/OpenVSP-3.56.0-win64 --out runs/verification/current_vspaero
```

四构型对照在最后一条命令中显式加 `--case rect --case elliptic --case swept30 --case bwb`。

## 历史记录

旧求解器的冻结输出、移植核对及 `face` 离散对照见
[历史离散化对照](archive/notes/DISCRETIZATION_COMPARISONS.md)。旧回放可在 `93abddb` 取回。
