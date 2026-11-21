# 文档索引

`docs/` 顶层只放**人读的定义文档**。过程材料（实验日志、移植清单、协议草稿、原始数据）不在
顶层：日志进 `archive/`，测试重放数据进 `verification/`。

    docs/
    ├── README.md            这一页（索引与维护规则）
    ├── MODEL.md             定义：求解对象、方程、离散、闭合、恢复、力、可微
    ├── NUMERICS.md          定义：装配、约束降维、求解器、判据、条件数、成本、自由尾迹
    ├── INTERFACE.md         契约：目录职责、调用链、默认值、输出字段、异常、数据、测试
    ├── VALIDATION.md        物理对照、数值验证与复现
    ├── DEFECTS.md           登记：#77/#92 的规格修改、已知缺陷与开放 issue
    ├── RETIRED_RESEARCH.md  退役代码与迁移文档清单
    ├── reference/           15 篇分模块参考（验收、标定、网格、尾迹、可微、VSP）
    └── archive/             历史：pr/ issue/ notes/ root/ evidence/ + 旧数据包

## 文档集合

| 文档 | 定义的内容 |
|---|---|
| [MODEL.md](MODEL.md) | 适用域；符号表；边值问题；P1 离散（壁面方程、尾迹势跳变）；两种后缘闭合（pressure / potential_jump）的数学定义；现役速度恢复（area / quadratic）的位置与算子定义；力与积分量（CL、CD_pressure、CL_γ、Trefftz CDi、e、Σ\|ΔΓ\|）；可微链定义；约定与不变量 |
| [NUMERICS.md](NUMERICS.md) | 算子装配（解析核、Duffy 求积、质量矩阵）；约束与零空间降维；pressure 的结构化 Gauss-Newton 与接受判据；potential_jump 的仿射+Newton 与多根性；回退规则；灾变门阈值；条件数事实；成本与线程；自由尾迹迭代；进程级约束 |
| [INTERFACE.md](INTERFACE.md) | 目录职责；生产调用链与函数签名；按入口的默认值表（壁面/闭合/恢复）；输出字段契约与残差定义；异常语义；Web 参数表；验证数据位置；测试契约；仓库约定 |
| [VALIDATION.md](VALIDATION.md) | 微观算法测试和宏观无粘气动对照的复现入口 |
| [DEFECTS.md](DEFECTS.md) | 缺陷登记格式；#77 两条（Γ 支撑、接触迹分裂势）、#92 移植；face+pj 系统性不一致；默认值与 docstring 不一致；研究/生产口径分叉；开放 issue #113/#115/#103 |

状态不在这五篇里：当前主线、测试基线、开放 PR、下一步只写 [STATUS.md](../STATUS.md)。文档与
代码冲突时以代码为准，并把该处登记为缺陷（[DEFECTS.md](DEFECTS.md) D-5/D-6）。

## 参考层（reference/，15 篇）

按主题查，不承载路线结论；每篇只负责一个模块或一次专项。

| 主题 | 文档 |
|---|---|
| 验收与标定 | [ACCEPTANCE.md](reference/ACCEPTANCE.md)（同一工况并排比较）、[CALIBRATION.md](reference/CALIBRATION.md)（解析与升力线标模）、[ERROR_SEPARATION.md](reference/ERROR_SEPARATION.md)（四类误差，2026-10-25）、[TEST_STATUS.md](reference/TEST_STATUS.md)（2026-11-03 基线） |
| 几何与网格 | [GEOMETRY_ROBUSTNESS.md](reference/GEOMETRY_ROBUSTNESS.md)、[PANEL_ROBUSTNESS.md](reference/PANEL_ROBUSTNESS.md)（37/37、180/180）、[MESH_CONVERGENCE.md](reference/MESH_CONVERGENCE.md)、[MESH_FEATURE_SIZE.md](reference/MESH_FEATURE_SIZE.md)、[MESH_QUALITY_GATES.md](reference/MESH_QUALITY_GATES.md)（#42，可选补丁） |
| 尾迹 | [ARCHITECTURE_WAKE_MODELS.md](reference/ARCHITECTURE_WAKE_MODELS.md)（三轴、材料线推进、B3、合并边界）、[HALF_MODEL_SCALING.md](reference/HALF_MODEL_SCALING.md)（LU vs GMRES） |
| 可微与力 | [DIFFERENTIABLE_MEMORY.md](reference/DIFFERENTIABLE_MEMORY.md)（15,273 → 1,810 MB）、[GEOMETRY_DIFFERENTIABILITY.md](reference/GEOMETRY_DIFFERENTIABILITY.md)、[EXACT_TREFFTZ_ENERGY.md](reference/EXACT_TREFFTZ_ENERGY.md)（#49，生产 CDi 的来源） |
| 外部 | [VSPAPI.md](reference/VSPAPI.md)（`vspapi/`，唯一 OpenVSP 客户端；VSPAERO 不是真值） |

## 证据日志（archive/，按时间追加，不作结论引用）

| 文档 | 性质 |
|---|---|
| [KUTTA_CONDITION.md](archive/notes/KUTTA_CONDITION.md) | 814 行后缘闭合实验日志（线性闭包 vs ITKUT、五层网格、恢复 A/B、平面形状对比） |
| [P1_G2_COUPLED.md](archive/notes/P1_G2_COUPLED.md) | 部分过时（"NumPy/SciPy CPU""基座 G1 已退役"），G2 耦合实测数字仍可用 |
| [ISSUE92_TORCH_PORT_MAP.md](archive/issue/ISSUE92_TORCH_PORT_MAP.md) | 移植 AST 清单与 S1–S12 切片 |
| [ISSUE92_B3_VSPAERO_PROTOCOL.md](archive/issue/ISSUE92_B3_VSPAERO_PROTOCOL.md) | 自由尾迹/B3 与 VSPAERO 的对照协议（600 行） |

## 归档与数据

- `archive/pr/`（19 篇）、`archive/issue/`（53 篇）、`archive/notes/`（35 篇）、
  `archive/evidence/`（退役研究线原始数据）；结论在正文，数据在同名子目录。
- 测试重放数据不在 `docs/`：`verification/issue92/`（s1–s9 冻结包与 `fixtures/`）、
  `verification/issue77/`。旧输出回放记录见 [历史离散化对照](archive/notes/DISCRETIZATION_COMPARISONS.md)。
- 退役代码：[RETIRED_RESEARCH.md](RETIRED_RESEARCH.md)（S12 清单、tag `research-p1-line-archive`）。

## 维护规则

1. **一篇一个模块一个入口。** 新结论只写进这五篇之一；过程材料进 `archive/`。
2. **文档只写定义与参照值，不写状态。** "现在是多少"写 `STATUS.md`；"定义是什么"写这里。
3. **每个数字带口径。** 参照表必须带 (壁面, 恢复, 闭合) 与网格规模，否则不可比（D-6）。
4. **复现命令。** 统一入口见 [VALIDATION.md](VALIDATION.md)。
5. **被取代的文档归档**，并在 `RETIRED_RESEARCH.md` 记一行（原路径、新路径、原因）。
6. **文档不承载测试数据。** 测试只读 `json`/`npz`，不从 Markdown 正文抓字符串当契约。

## 仓库约定

- 仓库根 `D:/3D/Projects/panel`；远程 `Stevenkerman27/Dpanel`，主分支 `master`。
- 不入库：`runs/`、`audit/`、`lifting3d/results/*.npz`、`papers/`。`runs/` 按功能分层
  （`cases/`、`benchmarks/`、`studies/`、`verification/`、`scratch/`），复现入口写在文档的
  复算一节。
- 单一环境：`C:/Users/zyx20/anaconda3/envs/myml/python.exe`；OpenVSP 用 `PYTHONPATH` 指向
  安装目录，不装包、不建第二个环境。
- 模块入口：`panel/README.md`、`torch_panel/README.md`、`monitor/README.md`、
  `test/README.md`、`lifting3d/README.md`、`potential/README.md`。
