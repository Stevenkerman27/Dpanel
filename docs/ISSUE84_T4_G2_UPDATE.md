# T4：G2 几何的一次真实几何更新（生产 audit 配置）

本文是 PR #91 的 T4 实验报告。目标是在 **G2 真实 CAD 半模**（runs/half_cad_wingbody_v1/surface.json，872 壁面 / 794 尾迹三角）上用**生产默认配置**（gate_mode=audit、velocity_method=analytic_edges）跑出 ① iteration 0 的初始解 ② 至少一次真实的几何更新，并给出每条结论的可复现证据。

所有数字都来自本仓库 docs/issue84/ 下的 JSON（工具在 tools/issue84/）。

## 1. 结论摘要

1. **iteration 0 正常**：audit 与 enforce 的载荷逐位相同（门控只决定哪些节点被推进，不参与壁面解）。main_wing 计数 root 90 / eligible 378 / audit_rejected 102 / rejected 0。
2. **生产配置下第一次几何更新被 wall 门拒绝**，且原因不是回溯预算：step 门在 8 次回溯后已通过，之后 13 次试探全部以 wall 失败（relaxation 缩到 1.4e-7 仍然失败）。
3. **wall 拒绝是假阳性**：命中的是尾迹三角 293（节点 [0, 56, 61]），它与壁面 13/18 **共用一个顶点**（节点 0 与 56 就落在机翼后缘顶点上）。扫掠凸包包含这个接触顶点，SAT 就报“碰撞”。减小位移不改变接触，所以回溯永远不收敛。
4. **把声明式接触对排除后更新可被接受**：1046 对（208 对共享边 + 838 对只共享一个顶点）→ 6 次更新全部 accepted。但 relaxation 立刻塌到 7.32e-5（11 次回溯），map_residual_chord 从 22.98 涨到 52.59 后走平，CL 从 0.4413554 只变到 0.4413881（+7.5e-5，相对 1.7e-4）⇒ **“accepted” 不等于“推进”**，材料尾迹没有向它自己的目标走。
5. **只排除共享边的对（208 对）不够**：iteration 0 仍被三角 293 与壁面 18 挡下，两者只共享一个顶点 ⇒ 需要排除的是**接触**（共享任意顶点），不是**碰撞**（有正穿透深度）。
6. fan 求积在这个实验里不产生新信息：iteration 0 的壁面解不依赖尾迹速度，g02 上实测三配置 iteration 0 的 CL 完全一致（0.16093316）；而一次 fan 28:8 采样轮的成本是 98 min（9496 targets × 0.618 s），所以 T4 不重复跑 fan。

## 2. iteration 0

pressure Kutta（生产默认），docs/issue84/t4_g2_analytic_audit.json，14.06 s：

| 量 | 值 |
| --- | --- |
| status | blocked_initial_nonlinear_solve（PR #87 的舍入级抛光失败，载荷可用） |
| CL | 0.43406115014632407 |
| CL_gamma | 0.43843173700189036 |
| CD_pressure | 0.006357145692946399 |
| CM_y | −0.11965433057432534 |
| pressure_kutta_rms | 9.0265e−07 |
| wall_rms | 1.21696e−03 |
| map_residual_chord | 22.0546 |
| main_wing 计数 | root 90 / eligible 378 / convergence 102 / audit_rejected 102 / rejected 0 |
| raised_tail 计数 | root 5 / eligible 15 / audit_rejected 0 / rejected 0 |

signed_speed Kutta（9.7 s）：CL 0.4413553954553846、CD_pressure 0.007126885631336986、convergence 83。

**audit 与 enforce 的载荷逐位相同**：门控只影响 proposed 几何，iteration 0 的载荷来自固定尾迹解，因此两者必然一致；变化的只有 counters（enforce 时 convergence 节点被冻结，rejected 计数非 0）。

## 3. 生产配置（audit）下的几何更新：被 wall 门挡住

| 运行 | 回溯预算 | 结果 |
| --- | --- | --- |
| audit，signed_speed，iterations=1 | 8 | block：8 次 step + 1 次 wall |
| audit，signed_speed，iterations=3 | 20 | block：8 次 step + 13 次 wall，relaxation=null |
| enforce，signed_speed，iterations=2 | 12 | 第 8 次回溯被接受，relaxation 5.859375e−04，moved 310，status=iteration_limit |

三档的证据分别在 docs/issue84/t4_g2_signed_audit.json 与 enforce A/B 的运行记录里。enforce 能推进的唯一原因是它把 83 个节点冻结，其中包含节点 56；节点 56 正是把三角 293 推穿后缘的那个节点（见 §4）。

### 3.1 step 门只是预算问题

tools/issue84/t4_step_probe.py 在 4 次回溯内记录违反量：main_wing 违反节点数 211 → 162 → 108 → 61 → 35，最大比值 173.7 → 86.8 → 43.4 → 21.7 → 10.86，节点局部尺度中位 0.05，零尺度节点 0，raised_tail 恒 0。按每回溯减半，第 8 次即降到 1 以下 ⇒ 与 sweep 行为一致。证据：docs/issue84/t4_step_probe.json。

### 3.2 wall 门是假阳性

tools/issue84/t4_wall_probe.py 记录每次 wall 失败的命中对：**13 次试探全部是同一个对**——main_wing 尾迹三角 293（节点 [0, 56, 61]，质心 (1.530307862978046, 0.0864906748843373, 0.0)）对壁面三角 13；平均位移从 −1.19e−05 一路减到 −7.43e−07 仍然命中（docs/issue84/t4_wall_probe.json）。

tools/issue84/t4_static_probe.py 给出几何事实：

* 未改动的参考几何 → geometry_failure 返回 None（不误报）；
* 只把三角 293 的三个节点沿 y 平移 1e−12 … 1e−03 → 全部 None；
* 但 old 三角 293 = [[1.41663, 0.1, 0], [1.470916, 0.059472, 0], [1.703378, 0.1, 0]] 与壁面 13 = [[1.26848, 0, −0.163462], [1.41663, 0.1, 0], [1.470916, 0.059472, 0]] **共享两个顶点**，直接调 convex_overlap(old, wall) 返回 True。

机理：panel/p1_coupled_material_wake.py:512 的 wall 段把「旧 ∪ 新」6 点扫掠凸包与壁面三角做 SAT，tol = tolerance_fraction(1e−10) × reference_chord ≈ 5.5e−11，凸包先按 1e−08 向质心收缩。尾迹片的根节点本来就在机翼后缘上（声明式连接），所以扫掠凸包必然包含这个接触点；SAT 无法把“点接触”和“穿透”区分开，于是判为碰撞。回溯只缩位移，接触不消失，只有当位移小于 1e−08 的收缩量时才会通过——那等于空步。

## 4. 正向实验：排除声明式接触对

tools/issue84/t4_declared_sequence.py 用同一份 G2 网格、同一套门限，只把**在参考几何上共享顶点的 wake/wall 对**从 wall 测试里排除（配对在参考几何上一次性算出，之后按 (group, wake_triangle, wall_triangle) 拓扑传递，不用坐标匹配——用坐标匹配会在第一次更新后失效）。

| min_shared | 排除对数 | 结果 |
| --- | --- | --- |
| 1（共享任意顶点） | 1046（208 边 + 838 单顶点） | 6 次更新全部 accepted，status=iteration_limit |
| 2（只共享边） | 208 | iteration 0 即 blocked，命中三角 293 对壁面 18（只共享 1 个顶点） |

min_shared=1 的六步序列（docs/issue84/t4_declared_sequence.json，68.8 s）：

| iter | CL | CD_pressure | relaxation | moved | map_residual_chord |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.4413554 | 0.0071269 | 5.859375e−04 | 393 | 22.982 |
| 1 | 0.4413646 | 0.0071310 | 7.324219e−05 | 393 | 52.244 |
| 2 | 0.4413669 | 0.0071338 | 7.324219e−05 | 393 | 52.408 |
| 3 | 0.4413730 | 0.0071384 | 7.324219e−05 | 393 | 52.493 |
| 4 | 0.4413787 | 0.0071432 | 7.324219e−05 | 393 | 52.541 |
| 5 | 0.4413842 | 0.0071481 | 7.324219e−05 | 393 | 52.570 |
| 6（无更新） | 0.4413881 | 0.0071529 | — | — | 52.587 |

读数：

* 第一次更新用掉 8 次回溯（relaxation 5.86e−04），之后每次都顶到 11 次回溯（7.32e−05），说明**全局回溯因子被最坏节点支配**，实际步长只有 7e−05 弦长量级；
* map_residual_chord 从 22.98 涨到 52.59 后走平 ⇒ 目标速度场与尾迹几何的失配在扩大后停滞，没有进入收敛通道；
* CL 六步只变 7.5e−05（相对 1.7e−04），CD_p 变 3.6e−05 ⇒ 系数已到不动点，但那是**限幅器**造成的不动点，不是几何收敛。

## 5. 结论与建议

1. **wall 门必须区分接触与穿透**：现在用扫掠凸包 SAT，接触（共享顶点/边）与穿透同判。正确做法是要求沿最小平移轴的正穿透深度超过 tol，或者在门控里显式携带声明式接触对（参考几何上算好、按拓扑传递）。这是 G2 上 audit 无法推进的直接原因。
2. **step 门需要逐节点尺度而非全局回溯因子**：单节点局部尺度最小 ~1e−03，最坏位移比值 173，全局减半要 8 次才能过；一旦最坏节点与别的限幅器耦合，回溯就停在一个无效的极小步。应改成按节点做截断（clip 每节点位移到 max_step_fraction × 该节点尺度），再对整片做一致性检查。
3. **不要用「accepted」当进度**：本次实验里 6 次 accepted 的净效果是 CL +7.5e−05、map_residual 平台在 52.6。T4 的验收量应是 map_residual 与尾迹形状变化，不是 accepted 计数。
4. fan 求积与本次结论无关：iteration 0 载荷与速度核无关（壁面解不读尾迹速度），实测 g02 三配置 iteration 0 的 CL 逐位一致；fan 只在推进阶段起作用，而推进阶段本次被门控挡住，跑 fan 只会把同样的墙重演一遍（且一轮 98 min）。

## 6. 复现

命令（工作目录 runs/fix_master，PYTHONPATH 指向该目录）：

    python -m tools.issue84.wake_ledger --main-half-surface D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json --case g2 --alpha 5 --kutta pressure --iterations 0 --gate-mode audit --out docs/issue84/t4_g2_analytic_audit.json
    python -m tools.issue84.wake_ledger --main-half-surface D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json --case g2 --alpha 5 --kutta signed_speed --iterations 3 --max-backtracks 20 --gate-mode audit --out docs/issue84/t4_g2_signed_audit.json
    python -m tools.issue84.t4_step_probe
    python -m tools.issue84.t4_wall_probe
    python -m tools.issue84.t4_static_probe
    python -m tools.issue84.t4_declared_sequence 1 docs/issue84/t4_declared_sequence.json
    python -m tools.issue84.t4_declared_sequence 2 docs/issue84/t4_declared_sequence_edge.json

产物：docs/issue84/{t4_g2_analytic_audit,t4_g2_signed_audit,t4_step_probe,t4_wall_probe,t4_declared_contact,t4_declared_sequence,t4_declared_sequence_edge}.json。
