# 节点坐标到原始面力/力矩的可微接口

状态：实验性 API，不替换 `panel/solve.py`。基线为 `local/panel-jax-20261025` 的 `08215a6`。本轮完成的不是通用 CAD 自动微分，而是**固定合法拓扑内，节点/环量站位/工况/参考量 → 面元方程 → 局部压力载荷**的可导入 JAX 函数。

## 参考已有实现后作出的选择

| 参考 | 本次采用的机制 | 没有照搬的能力/假设 |
|---|---|---|
| VortexAD `pm_class.py` 的 points / cell_point_indices / TE 索引分离 | 数值几何与固定连接关系分开，公开连续节点输入 | 不复制默认 Cp 截断、不绕回上游不适配机身接触的尾迹 |
| FLOWPanel `solve_ludiv!` + ImplicitAD | 同一 LU 用于正向和转置伴随；对解的隐式方程求导 | 不宣称 GMRES 必然更快，不把 dense matvec 叫 matrix-free |
| JAX `lax.custom_linear_solve` | 可微 matvec、显式 transpose_solve，`symmetric=False` | 不通过停止梯度的 LU 因子来丢弃 A(X) 的导数 |
| OpenAeroStruct 几何模块 | 固定构型族内连续变形，网格分辨率与设计变量数量独立 | 不声称重网格、布尔和接触拓扑变化可微 |

来源（2026-10-25 核对）：
- https://github.com/LSDOlab/VortexAD/blob/ab6493981d5ff19721ba9fc42b6d015310d3d5cd/VortexAD/core/pm_class.py
- https://flow.byu.edu/FLOWPanel.jl/stable/examples/sweptwing-solver/
- https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html
- https://mdolab-openaerostruct.readthedocs-hosted.com/en/latest/advanced_features/geometry_manipulation.html

## 新代码

`panel/differentiable.py` 暴露：

```python
inputs, metadata = load_prepared_run("runs/EXISTING_CASE/solution")
validate_configuration(inputs["vertices"], inputs["station_y"], inputs["topology"])
fields = jax.jit(solve_fields)(**inputs)

# 上游可给出任意局部载荷种子，不构造完整 N面力×V坐标 Jacobian。
def local_loads(X, Y):
    out = solve_fields(**{**inputs, "vertices": X, "station_y": Y})
    return out["panel_forces"], out["panel_moments"]
values, pullback = jax.vjp(local_loads, inputs["vertices"], inputs["station_y"])
dJ_dX, dJ_dY = pullback((force_seed, moment_seed))
```

`load_prepared_run` 仅在图外初始化：读取先前通过几何校验的 CLI 产物，复用真实 cut-aware 邻接、面片和尾缘配对。不重新划网格，也不重新跑求解。它是过渡装载器，**不是已完成的 GRASS 参数化到公共外表面的生成器**。

`solve_fields` 没有文件 I/O、CLI、副作用或 NumPy 转换。每次根据当前输入重新计算面积/法向、源强、双层势、尾迹一次矩、线性势跳跃基、TE 高斯积分长度与基函数、表面梯度、原始 Cp 和局部力矩。离散 `wake_interval/te_interval/hat_map` 固定；其内的基函数值与 slope 不被冻结。

`station_y` 是连续输入，必须由同一几何参数化与节点一起移动。只移动旧网格而冻结站位、或任意把节点移出原条带，不属于合法梯度。`validate_configuration` 在图外拒绝非平面尾迹、错误绕向、条带错位和退化模板；它不替代 CAD/wall–wake 相交的完整独立审计。较大形变仍须重新执行现有几何审计。

## 单位与返回值

所有输入节点/站位/力矩参考点使用相同物理长度单位；`freestream`、`density`、参考面积/弦长也显式输入。面力用 `q=.5*rho*|U|^2` 乘原始 Cp 与面积；长度用 m、速度 m/s、密度 kg/m³ 时，输出是 N 与 N·m。组件数组按标签 10、20、30 分组，并非任意多翼实例 ID。

既有 CLI 在归一化坐标中使用 `rho=2,V=1,q=1`。装载器将坐标换回物理长度；回归时逐面力/力矩需分别与旧数组乘 `chord²/chord³` 后比较，mu/gamma 则乘 chord。参考面积/弦/点本身可作为求导输入；默认装载值固定来自该 run，设计中是否让它们随几何变化由调用方显式决定。

## 线性求解与反向

保持原增广系统 `[D W; -T C] [mu; gamma]=[S sigma; 0]`，压力仍用原来的无权仿射重构。`linear_solve` 在每次前向做一次稠密 LU，利用 `custom_linear_solve` 定义隐式梯度，并复用因子做转置回代。`stop_gradient` 仅作用在数值分解路径，`matvec=lambda x: A @ x` 仍含完整 A 的几何导数。

没有跨不同几何缓存旧矩阵，没有 FMM，没有提高物理阶次，也没有改善原 Cp 精度的声明。`physical_accuracy_accepted=false` 保持不变。

## 本次环境已经运行的检查

CPU、Python 3.13.5、JAX 0.9.0.1、NumPy 2.3.5、float64，6 项测试通过（25.13 秒）。几何是程序生成的**有厚度封闭翼＋有限尾迹**：40 个壁面三角形、8 个尾迹三角形、43 个状态未知量；不是用户的完整翼身 CAD。

- 原始局部力/矩、部件/总力矩闭合及真实增广残差；
- 任意局部力/矩种子对顶点、同时移动站位的 VJP；
- 单个合法壁面顶点 z 扰动对照；
- 密度/速度、坐标缩放、参考面积/力矩点的值及梯度规律；
- 非平面尾迹与脱离冻结条带的变形拒绝；
- 从旧 `solve_model` 提取函数 AST，在同一网格、基准与 bulge/span 形变下逐项比较 A、rhs、mu、gamma、Cp、逐面力矩（rtol/atol 2e-11）。本次容器使用从已读取源码转录的原函数段，没有执行完整旧 CLI；本地验收应读取仓库中原文件。

随机局部载荷目标的一个合法联合几何方向：AD=0.019217626145539755；中心差分 h=1e-3/1e-4/1e-5 的相对差分别为 3.21e-10、4.19e-11、1.18e-9。真实翼身回归、GPU、吞吐和 GRASS 链式 VJP 尚未运行，不能转写为通过。

## 本地 agent：不要重复 CAD，只用已有保存场

```bash
python -m pytest -q test/test_differentiable_panel_fields.py
python -m panel.verify_differentiable \
  --run runs/EXISTING_CASE/solution --out runs/NEW_VERTEX_CHECK --derivatives
```

先用单翼/前置机身的小网格，再用跨后缘和粗双机身；每个输出目录必须新建。驱动会比较保存的原始 Cp、全部面力/面矩、mu/gamma、CL/CM，检查真实残差，再对任意局部种子做非均匀展向/后掠/厚度联合形变的三档差分。默认拒绝超过 2000 个面元，避免不必要的稠密 AD 内存风险。

回报 SHA、JAX/BLAS/设备、冷/热时间、实际峰值内存、逐字段差和 VJP 误差；若失败保留 `report.json`/traceback，不改变容差或压力恢复方法。

## 明确剩余工作

1. 已验证的真实翼身 run 上完成数值回归与 VJP 后，才重接现有 CLI/公共 API；本 PR 不改 `aero.py` 或 `panel/solve.py`。
2. 将 GRASS 的 CST/机身/连接区参数变成兼容的 X(theta)、station_y(theta)；固定拓扑合法范围外必须离散重建，不能通过 autograd 魔法处理布尔操作。
3. 完成 Torch→JAX→Torch 的局部力 VJP 桥及随机设计方向的整体差分，不在这次代码中声称桥接已交付。
4. 压力物理精度、黏性闭合与训练可信域仍需独立验收。前一步的正确离散导数不能替代这些门槛。
