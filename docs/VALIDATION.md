# 验证与参照值

| 层级 | 依据 |
|---|---|
| 实验验证 | 用相同几何、迎角、雷诺数、马赫数和转捩条件，对照风洞或飞行数据 |
| 理论校准 | 用解析流动检查精确极限，用升力线等理论估计检查适用范围内的量级 |
| 数值验证 | 用网格细化、方程残差、约束和中心差分检查当前离散的解和梯度 |
| 外部求解器对照 | 对齐几何、工况和积分口径后比较预测差异 |

现役标模入口见 [CALIBRATION.md](reference/CALIBRATION.md)，数据在 `benchmarks/calibration/reference.json`。
球体对照解析 \(C_p\)，薄翼和椭圆翼对照升力线理论。NACA 0012 的升阻和压力实验数据来自不同工况，
校准程序分别登记其来源和条件。求解与实验对接状态见 [STATUS.md](../STATUS.md)。

二维圆柱和 Joukowski 对照分别见 [全势流圆柱结果](../potential/RESULTS.md)和
[P2 翼型结果](../airfoil2d/RESULTS.md)。

矩形翼、30° 后掠翼、椭圆翼和翼身融合体的独立求解器对照见
[VSPAERO 对照记录](archive/issue/ISSUE92_B3_VSPAERO_PROTOCOL.md)。

## 复现

```powershell
python -m tools.calibration --out runs/verification/calibration
python -m tools.calibration --case C05 --refine --out runs/verification/calibration_refine
python -m pytest -q -p no:cacheprovider --tb=short test
```

## 历史记录

旧求解器的冻结输出、移植核对及 `face` 离散对照见
[历史离散化对照](archive/notes/DISCRETIZATION_COMPARISONS.md)。旧回放可在 `93abddb` 取回。
