# 多独立尾迹弱Kutta块：第一阶段数值原型（Issue #41）

这是用户要求“一片独立升力面一张涡片”的 **Step B 研究代码第一层**。前置的 PR #40 是v2 schema文档；本PR在该契约分支上增加**真实数值块组装与单个TE的精确弱式积分**。暂不接真实CAD、双升力面JAX势核、GRASS训练/默认路径。

对G组、同一湿壁面 mu 与各组 Gamma，有：
```text
[D, W1, W2,...]
[-T1, M1, 0]
[-T2, 0, M2] ...
```
右端 [S*sigma,0,...]。单组退化为v1块形；零组退化成仅湿壁面D。组间**通过 mu 耦合**，即删除一组会通常改变其它状态。条件下的正确限制实验：把被删组的已求得影响 `Wh*gamma_h` 转为保留系统的已知RHS强迫，应复现完整解。

原 `half_native.py` 用2点Gauss积 `mass_ij=sum w B_i B_j` 和 `integrals_ei=sum_q w B_i`；本PR `te_weak_blocks` 精确复现 `T=(integrals.T@incidence)/measure[:,None]`，`M=mass/measure[:,None]`。特别验证 `mass != integrals.T@integrals`，纠正 PR #40 草案错误。

云端隔离 Python3.13/NumPy2.3.5/pytest9：**10 passed**（无skip）。包括非等式的M反例、零组、单组矩阵、两组强耦合与删除反例、转置伴随/有限差分、支持度和非法输入拒绝。没有在本环境运行完整DPanel/JAX/Gmsh集成，不宣称双尾迹气动物理解已完成。

本地下一步按 #41：
1. 在独立PR复用本 `assemble_shed_groups/te_weak_blocks` 的可解释结构，对接 `panel.kernel.triangle_potential` 为**每组实际水平wake**分别计算 `W_g`，加 v2 `shed_groups.py` 真实边/bank/wake/DOF 拒绝规则；`G=1` 必须严格复现现有原生求解器，`G=2` 用两高度厚翼小夹具验证，不可自行生成伪TE；`G=0` 是纯闭合机身。
2. 之后另一个独立PR扩展真正 OCC 正半域两尾迹共形 `fragment`，保留 `v1` default；固定conv CAD 新目录验证独立Gamma/Kutta/平尾载荷。
3. 融合后的几何精度需要同几何VSPAERO或CFD外部基准。适应性网格研究在#42/PR43，**不要同时改物理模型与网格**。

无新增依赖。保留 `physical_accuracy_accepted=false`，不合并、不启用正式训练。
