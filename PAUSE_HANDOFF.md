# 当前交接：2026-10-26

**用户已恢复本地目标，本轮推进尾迹和计算准确性；旧暂停指令不再约束本地研究。** 2026-10-26 15:00 用户恢复云端协同与 GitHub 监听（15 min 轮询，代理 `http://192.168.3.9:7890`）；本地进展已推至 `local/panel-jax-20261025`（commit `45bf2e6`），并在 issue #25 / PR #26 回报。代理失效就告知，不排查或直连。

最新方法、准确性证据、失败和复现命令见 [WAKE_ACCURACY.md](docs/WAKE_ACCURACY.md)。本轮实际入口为根目录 `panel/solve.py`；半模、矩阵无关和后续伴随主要在隔离 PR 工作副本，未集成入口不能混用。现有 kernel 分块/checkpoint 已保留。

- 18组同壁面方向对照、45项实现检查通过；五例固定尾迹对历史结果最大差约9e-14。三个处方尾迹AD批次135标量比较通过。
- 任意法向/物质坐标线性矩及速度的11项独立核检查通过；增加真实尾迹总流切向、物质环量和压力连续性审计。
- 参考锁定VSPAERO源码实现局部总流驱动物质尾迹研究脚本，保留根部、环量标签和全部迭代；支持三维横向运动/卷曲，尚非物理验收。
- 单翼5°线更新10步近尾迹法向RMS .06549→.01647，但近尾迹Cp跳跃RMS .003005→.009412；面速度最小二乘8步进一步得到法向RMS .009735、压力跳跃RMS .010955，仍未接受。壁面误差基本不变，下一步联合处理尾迹运动学、环量输运/压力连续性及场/形状非线性耦合。
- 近相切radius8复核已完成，原始证据在 `runs/panel_error_budget_v2/`；下文“尚未复核”属于旧状态。
- GitHub：Issue25/草稿PR26已读，head `664d9381686d8cfe1070debba8f9a0e327cbdc51` 已fetch为 `origin/pr26-research`；2026-10-26 14:57:52末次轮询无新事件，15个开放PR，主分支未合并后续研究。

默认 `fixed` 尾迹未替换。全接触尾迹倾斜仍须重建CAD流体切面；`far-freestream`只改变远段。冻结偏移的AD不是自由尾迹全导数。`physical_accuracy_accepted=false`保持不变。NCPU=8，全部输出使用新目录。

## 历史暂停交接：2026-10-25

以下保留历史记录，其状态和待办以本页顶部与最新尾迹报告为准。

**当前状态：已按用户要求收尾并暂停。完整鲁棒求解器目标尚未完成，恢复必须由用户明确提出。** 本地子任务已收尾；云端协同与 `Dpanel GitHub只读监听` 自动任务均暂停。没有提交、push、merge或对外评论。

## 恢复先读

- 最新方法、构型表及保留误差：[PANEL_ROBUSTNESS.md](docs/PANEL_ROBUSTNESS.md)。
- 紧凑证据及原始字段哈希：[summary.json](docs/panel_robustness_evidence_v2/summary.json)。
- 实际入口及依赖：[panel/README.md](panel/README.md)。
- 路线和总体状态：[SOLVER_ROUTE.md](SOLVER_ROUTE.md)、[DEVELOPMENT.md](DEVELOPMENT.md)。
- 前阶段VortexAD闭环：[WINGBODY_PANEL_LOOP.md](docs/WINGBODY_PANEL_LOOP.md)。

用户优先级仍是几何鲁棒性、速度、方法可解释性；不优先追逐局部精度，但必须如实报告原始误差。研究脚本保持线性、命令式，先完成后完善。NCPU请求8。禁止用压力裁剪、隐藏正则化、SVD截断、涡核或假后缘掩盖故障。

## 本轮已完成

新增独立 `panel/` JAX源/偶极子面元求解器，支持真实CAD公共外表面、机身跨后缘、带接触边界的流体侧尾迹、粗机身与双机身。VortexAD保留作前置机身参考；旧P2全势路径保留作独立对照，没有改写其历史失败。

1. `panel/kernel.py`：float64解析三角源/偶极势，自面内侧迹-1/2，解析线性wake一次矩；7项独立面积积分及几何导数检查通过。
2. `panel/cad.py`：复用OCC布尔与两遍接触站位，只生成表面网格；壁面按邻接一致绕序及正体积确定外向，不生成/优化体网格。双机身616/1646壁面，粗机身824壁面均成功。
3. `panel/build_surface.py`：读取旧P2或新surface-only NPZ，导出 `dpanel.planar_cut.v1`；保存真实TE、接触cut、流体wake及Gamma等价映射。公开夹具位于 `cases/panel_cut/`。
4. `panel/solve.py`：联合势方程、解析线性尾迹、真实外露TE弱势跳跃闭合、接触区仍未知的常量Gamma、无正则表面势重构、原始Cp及翼/身/接缝力和力矩。AD涵盖迎角、中心厚度与展长，固定拓扑。
5. 几何门禁：闭合/绕序/正体积，真实wall/wake交边与上下bank，TE条带积分和支持，接触图重新推导的约束，有向边相消与自由边界分类，面心绕数及wall-plane交线穿wake开内部检查。
6. `panel/audit.py`：独立NumPy/Numba核、真实外场速度、面内点、外侧趋近、接触/TE势跳跃及QR梯度审计。真实外场和切平面压力重构分开记录。
7. `aero.py` 新增显式 `panel-cut` 分支，`panel/run_case.py` 保存命令和manifest；`panel/export.py` 导出真实表面和wake的VTU，保留原始Cp与部件标签。旧入口沿原路径运行。

## 验证与产物

| 产物 | 已核实状态 |
|---|---|
| `runs/panel_robust_matrix_v1/summary.json` | 37/37项主检查；14个构型/工况解、3个尺度/参考点解、13个无效输入拒绝；四组AD/FD共180/180标量比较 |
| `runs/panel_topology_matrix_v1/summary.json` | 10/10；8个公开夹具接受，小闭体穿wake与悬空自由边界明确拒绝 |
| `runs/panel_kernel_v1/checks.json` | 7/7核与核导数检查 |
| `runs/panel_cli_two_final/` | 最终求解源码完成统一入口、45项导数及表面/wake VTU；CL=0.263212956 |
| `runs/panel_cli_cad_v1/` | 从CAD到表面网格、求解及VTU全流程通过 |
| `runs/panel_surface_cad_two_v1/`、`panel_surface_cad_two_fine_v1/` | 双机身粗/细表面CAD产物与完整日志 |
| `runs/panel_surface_cad_thick_v1/` | 半径0.6粗机身与NACA4412表面CAD产物 |
| `runs/panel_cad_limits_v1/` | 4个近相切CAD探针，只有aft=1.0001建网格成功；其余被farfield包围盒检查阻挡，尚未完成近相切验证 |

JIT后一次前向0.050～0.427秒，不含CAD、编译和进程启动。0.001/1000倍同拓扑尺度和平移后Cp最大差5.12e-12/3.91e-14；力矩参考点变换误差6.66e-16。不得据此宣称任意CAD、任意wake或实际八线程加速。

主矩阵使用新增交线门禁前的源码快照；补强门禁有独立10项验证，最终双机身入口另有完整导数和导出。`cad.py`最后只改了bounds阶段名和失败信息，尚未重跑CAD。不同源码哈希与批次保留，不伪称所有实验都由同一最终源码执行。

## 未完成与准确诊断

**物理精度全部仍为false。** 双机身粗/细CL为0.263213/0.312072，相差约18.6%；全域真实壁面法向RMS分别0.02846/0.04388 U，细化没有整体单调改善。跨后缘粗例法向峰值达3.61692 U。重构Cp有限、切平面速度法向为零及线性真残差约1e-15都不能冒充真实无穿透。

后缘下块是弱势跳跃闭合，不是直接速度Kutta条件；尚无独立尾缘速度精度验收。尾迹长度30Cref，截断模型显式保留。接触条件使用y投影和流向恒定Gamma，尚未认证重叠投影分量的独立环量、一般三维旋转、非平面/自由尾迹、自交壁面或跨CAD拓扑变化的导数。

近相切探针的三个首轮失败来自 **OCC保守包围盒超出radius=5**；旧failure记录的stage误写为TE验证阶段，traceback才是准确定位。当前代码已改为 `cad_bounds_pass_*` 并附包围盒/radius。不能将这些失败解释为近相切几何不可分辨或TE声明无效。尚未增大radius重试。aft=1.0001只有表面网格，接触展向长度0.00699991，未做面元求解。

少量 `.conda_env` 进程向 `cases/` 创建新文件时有PermissionError，Anaconda生成器和PowerShell文件复制可正常操作；本轮未改ACL，历史失败保留。`runs/`输出正常。旧6504壁面 `two_bodies.json`太密，恢复时优先用 `two_bodies_coarse.json`，不要直接拿它做稠密JAX基准。

## 恢复时的实际起点

以下是待办，暂停后不得自动执行：

1. 先以新目录、较大farfield（例如radius8）复核4个CADlimits输入，排除OCC保守包围盒干扰，再判断真实近相切故障；保留原radius5失败。
2. 对aft=1.0001的真实接触表面完成独立几何检查、面元解、原始场与载荷；不可仅凭生成网格计作全流程通过。
3. 继续扩大几何/接触矩阵并明确支持域；依用户优先级控制成本。进一步研究精度时，先分离表面几何变化、离散误差、势重构与尾缘条件，不靠裁剪或平滑。
4. 再决定是否推广非平面wake、接触图或更高阶表面势表示。保守全势/P2及WBCAero公开方法研究保留，整体反编译继续暂停。

## 复现环境与命令

工作目录 `D:\3D\Projects\panel`。主环境 `.conda_env/python.exe`：Python3.11.17、NumPy2.4.6、SciPy1.17.1、JAX/JAXlib0.10.2、threadpoolctl3.7.0；无Numba/Torch。独立审计 `C:/Users/zyx20/anaconda3/python.exe`：Python3.13.9、NumPy2.3.5、SciPy1.16.3、Numba0.62.1。本地Gmsh4.15.2位于 `audit/runtime/python/`。

```powershell
.\.conda_env\python.exe aero.py --case cases/panel_two_bodies.json --out runs/NEW_TWO --export-surface
.\.conda_env\python.exe aero.py --case cases/panel_cad_wingbody.json --out runs/NEW_CAD --export-surface
.\.conda_env\python.exe panel/verify.py --out runs/NEW_MATRIX --audit-python C:/Users/zyx20/anaconda3/python.exe
.\.conda_env\python.exe panel/verify_topology.py --out runs/NEW_TOPOLOGY
C:/Users/zyx20/anaconda3/python.exe panel/audit.py runs/NEW_TWO/solution --off-surface
C:/Users/zyx20/anaconda3/python.exe panel/report.py --matrix runs/NEW_MATRIX --topology runs/NEW_TOPOLOGY --out docs/NEW_EVIDENCE
```

所有输出使用新目录。工作区仍有大量先前未提交内容与本地环境，不要清理、覆盖或回退。没有为本次工作创建提交。公开JSON夹具可脱离历史体网格复现。

GitHub只能经 `http://192.168.3.9:7890`；代理失败停止并告知，不排查、不改直连。本轮只读轮询无新事件。自动任务ID `dpanel-github` 已置PAUSED；原15分钟配置与只读提示保留。恢复目标不等于授权push、评论或恢复云端协同。

---

## 历史交接原文：2026-10-24

以下保留旧阶段原文，不代表最新状态；独立原始归档见 `docs/PAUSE_HANDOFF_2026-10-24.md`。

总目标已按用户要求暂停，尚未完成整体验收。本次整理没有启动新实验或继续修改求解代码；检查时没有 Python 实验进程运行。恢复须由用户明确提出。

## 结论与入口

当前路线为保守全势主线、独立低速面元参考。用户最近确定的优先级是几何鲁棒性，其次才是局部压力精度；保持方法可解释，不用压力裁剪、SVD 截断、隐含正则化或指定根部升力换取稳定。

- 总体能力与历史验收：[DEVELOPMENT.md](DEVELOPMENT.md)。
- 几何状态总览：[lifting3d/GEOMETRY_STATUS.md](lifting3d/GEOMETRY_STATUS.md)。
- 最新 CAD 结果与失败：[lifting3d/CAD_ROBUSTNESS_RESULTS.md](lifting3d/CAD_ROBUSTNESS_RESULTS.md)。
- CAD 方法契约：[lifting3d/CAD_PROTOCOL.md](lifting3d/CAD_PROTOCOL.md)；验收范围：[CAD_ROBUSTNESS_PROTOCOL.md](lifting3d/CAD_ROBUSTNESS_PROTOCOL.md)。
- 路线选择和 WBCAero 研究边界：[SOLVER_ROUTE.md](SOLVER_ROUTE.md)、[WBCAERO_STUDY.md](WBCAERO_STUDY.md)。

## 已完成

前阶段参数化机翼/椭球机身共 24 个不同几何配置通过网格与有限流场检查，另有 18 个存储网格的独立尾迹基审计。分部件压力力、力矩和原始表面输出已实现。原有二维、闭体、Schur 对照和失败记录保留，详情见总览。

本阶段新增完整体网格升力 CAD 路径，保留弯度和上下不对称几何：

| 证据 | 已核实状态 |
|---|---|
| `lifting3d/results/cad_pipeline_v3/summary.json` | 6/6：对称翼、弯度翼、两种翼身、1000 倍尺度和平移、后掠渐缩弯度翼；均完成 mesh / audit / solve / loads / export |
| `lifting3d/results/cad_contacts_pipeline_v1/results.json` | 3/4：机身最大截面晚于后缘、机身完全在后缘前、半径 0.6 的粗机身通过；双机身首次网格失败 |
| `lifting3d/results/cad_invariance_v1/results.json` | 同一翼身网格，原尺度及 0.001/1000 倍缩放和平移均通过；Cp 最大差分别约 1.14e-9 / 2.93e-12 |
| `lifting3d/results/cad_input_checks_v1/results.json` | 错位、非平面、乱序和越界后缘 4/4 预期拒绝，均未输出接受网格 |
| `lifting3d/results/cad_contacts_repair_v1/two_bodies_curvature16/mesh.json` | 双机身开启曲率尺寸约束后建网格成功；只有 builder 内部检查，尚无独立审计和流场结果 |

因此最新 CAD 阶段有 **9 个配置跑通完整流程，另 1 个双机身配置只完成修复后的建网格**。这些与前阶段的参数化批次分开统计，不作为任意 CAD 认证。

## 本阶段实现

- `lifting3d/build_cad_mesh.py`：闭合 wing CAD、可选 body CAD、用户声明后缘；两遍构造接触站位，完整体网格，原生坐标输出与源码/输入哈希。
- `cad_cut.py`：按 cut 节点局部四面体星邻接分离两侧，不按全局单元中心 z 正负切分；零环量端点保留共享节点。
- `cad_contact.py`、`cad_basis.py`：同时读取曲面边界曲线和嵌入曲线；先排除声明后缘，再构造接触组件。各接触投影内 Gamma 为常量且仍未知，支持非中心与多个区间。
- `cad_components.py`、`cad_te.py`：由保留的 CAD 源面识别部件归属；检查后缘采样点是否落在机翼有限边界曲线上，避免将延长线误判为有效后缘。
- `solve.py`：CAD 分支复用 P2 全势方程，weak Kutta 仅沿声明后缘积分，符号来自拓扑 bank；旧参数化网格回归通过。
- `test_mesh.py`：检查完整外边界、P2 中点、尾迹成对关系、站位映射、条带相容性和每个环量自由度的真实后缘支持。
- `aero.py`：新增 `lifting-cad-p2`，CAD 路径相对 case 解析，机身可省略，沿用载荷和导出流程。已有网格审计在输出副本执行。
- `analyze.py`：一般 CAD 不再套用 NACA 前后缘区域假设。

试验驱动为 `run_cad_robustness.py`、`run_cad_input_checks.py`、`run_cad_invariance.py`。配置与几何输入在 `cases/cad_wingbody.json`、`results/cad_inputs_v1/`、`results/cad_contact_inputs_v1/`。

## 已知失败与解释

旧 CAD 接触识别只找曲面边界曲线，遗漏了机身上的嵌入曲线，使中央环量自由度没有后缘条件。现已补齐；旧网格及失败仍在 `results/cad_integration_v1/`。

审计脚本曾把所有 wall/wake 交边都计作后缘，并错误要求“有机身就必须有接触”。两条规则已纠正：只有声明后缘贡献 Kutta 支持；前置机身无接触是合法情况。纠正后的审计明确检出旧翼身网格的三个无支持自由度，记录在其 `audit_corrected.json`，没有覆盖旧结论。

双机身首次 Gmsh HighOrder 优化失败，子进程返回 3，原日志保留。曲率 16 重试生成 36720 节点、20874 单元，两个接触区间保持分离；不据此声称求解通过。曲率规则尚未成为通用自适应修复流程，默认参数也未因此修改。

## 恢复时的实际起点

以下只是待办清单，本次暂停后未执行：

1. 对 `cad_contacts_repair_v1/two_bodies_curvature16/mesh.npz` 做独立审计，再求解、积分和导出；核对两组环量合并保持独立。
2. 增强曲面 Tet10 正性检查。暂停前只提出 Gmsh 自适应 `minDetJac` 交叉检查，**尚未实现或测试**；当前仍是 165 点/单元抽查。
3. 基于双机身失败研究可解释的几何尺寸规则与优化失败诊断，保留原失败及曲率重试成本。
4. 扩展真实 CAD、近相切、多接触和资源边界；新增 CAD 的可压缩工况也尚未形成独立验收矩阵。
5. 更广泛的非平面后缘、复杂尾迹、一般三维旋转和最终物理验收仍未完成。局部压力加密按当前优先级暂缓。

现有新 CAD 流程限定：后缘位于归一化 z=0，严格按 y 递增、x(y) 分段线性；固定平面尾迹、weak Kutta、采样严格亚声速。接触曲线极值依赖有限参数扫描，不是任意样条的全局认证。

NCPU 默认请求 8；SuperLU 不据此声称八线程，PostView 网格分支仍用 1 线程。WBCAero 主控大函数整体反编译继续暂停。历史输入、日志和源码快照均保留；后续使用新输出目录。
