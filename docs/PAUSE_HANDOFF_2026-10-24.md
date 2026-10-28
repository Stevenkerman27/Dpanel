# 暂停交接：2026-10-24

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
