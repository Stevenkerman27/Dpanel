# 面元压力梯度重构独立诊断（Stage 1）

状态：**研究诊断，尚未运行实际保存字段，未通过物理精度验收**。基于 `local/panel-jax-20261025` 的已保存字段格式；本次 PR 不改变 `panel/solve.py`。

## 为什么做这个试验

独立实验 `docs/ERROR_SEPARATION.md` 报告：aft=1.0001 的粗、中、细接触网格 h=0.30/0.20/0.15，生产压力积分 CL 为 0.251697/0.303516/0.319645；环量积分 CL 为 0.329871/0.336903/0.339294。差值从 -23.4% 缩小到 -5.8%，但 `physical_accuracy_accepted=false`。该差异不能直接当作压力重构误差的严格分离，因为不同面元几何离散、势恢复与环量积分本身也会改变；当前研究只固定已保存 `mu`、几何和尾迹并改变后处理拟合，避免混淆主方程。

## 本 PR 比较两个基础候选，另提供可选的二阶试验

- `affine_qr`：使用**原来完全相同的**切平面位置差和 cut-aware 邻接、同一 `mu_j-mu_i`，用 QR 解未加权两列仿射最小二乘。它应在浮点误差内复现生产中 2×2 正规方程的梯度；如果显著不同，说明数值条件性或实现问题，而非物理改进。
- `weighted_affine_qr`：相同模型，但将每个邻点方程按切平面距离的倒数加权；不改变任何势变量或边界条件，不做 SVD 截断、正则化或截止压力。它仅是实验性候选，**不能预设会降低 CL 偏差**。
- 可选 `quadratic_qr`（CLI `--quadratic`）：严格求解五列局部二阶多项式的 QR（线性导数 2 项、二阶曲率 3 项），没有惩罚项或截断。只有至少 5 个足够独立的合法邻面才可用，欠秩时直接失败而非退回一阶；本次它也是**离线研究候选**，不是已验证的高阶气动面元。

对不满秩、非法邻接或非有限数据显式失败，不回退到默认法；候选计算不进入生产。QR 内部也不做高阶重构或额外的尾缘附近跨面拟合。

## 运行方式（无需 JAX/Gmsh）

在已有实际求解 `runs/CASE/solution` 上执行，所有结果写到新文件，不覆盖原场：

```bash
python -m pytest -q test/test_pressure_reconstruction_probe.py

python panel/probe_pressure_reconstruction.py \
  --run runs/EXISTING_CASE/solution \
  --out runs/pressure_probe_CASE.json
# 如果所有面元均满足五列二阶拟合满秩，也可另外在新路径测试：
python panel/probe_pressure_reconstruction.py \\
  --run runs/EXISTING_CASE/solution \\
  --out runs/pressure_probe_CASE_quad.json --quadratic
```

`--run` 必须包含求解器保存的 `fields.npz`、`surface_input.json`、`result.json`；指定新 `--out` 若存在直接报错。读取的量包括 `mu`、`panel_center`、`panel_x_dir`、`panel_y_dir`、`neighbor_indices`、`panel_area`、`panel_normal`、`Cp_raw`、`component_ids` 和 `gamma`。运行时不触碰 JAX 或 CAD。

输出记录：原生产 CL / CM_y、环量 CL（**不是外部物理真值**）、候选 CL / CM_y、压力 RMS 和局部面力 L2 差，局部切向模板奇异值比，以及变化最大的 5 个面元。候选 CL、CM_y 采用 `panel/solve.py` 同样的无量纲参考面积、坐标中心、法向和升力轴。

## 本地验证顺序

1. 先跑不依赖 JAX 的 5 项单测：已知二维线性势在局部坐标精确恢复；全共线模板必须报错；作用点与参考点移动后的压力力矩公式正确；环量站位积分正确。
2. 再对一个 `panel/verify.py` 中的较小已保存工况做工具测试。检查 `affine_qr` 与原 `Cp_raw` 的最大面元差是否接近舍入量；若不接近，先报问题，不得跳过。
3. 再对 `runs/panel_error_budget_v2/` 的 aft=1.0001、h=.30/.20/.15 三档场进行对照。记录面元 N、每组 wall_time、CL / CM_y、Cp_min/max、残差及异常面元。
4. 只允许用“改进/退化的对照数字”汇报候选；不能单凭压力 CL 靠近环量 CL 就宣布高精度，更不能把该结果写入默认压力后处理。
5. 只有后续真正的独立物理压力与网格收敛、结构局部载荷及设计参数 VJP 验收通过，才考虑修改生产求解器。GRASS 生成器仍不得默认使用 `physical_accuracy_accepted=false` 的字段作为高可靠性监督目标。

## 计算成本边界

读取一次已保存场后进行一次 O(N·k) 小型 QR 拟合，N 是壁面面元数、k 是 cut-aware 局部邻居数；本脚本本身不解稠密面元主方程，因此运行时间**不可当作完整气动预测的速度**。更高阶重构和快速 JAX VJP 属于后续单独设计，不能用这个离线脚本替代。
