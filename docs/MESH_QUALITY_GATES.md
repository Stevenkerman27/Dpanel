# 实际 OCC 网格预算与质量门禁（Issue #42）

这是 PR #43 已由本地真实 Gmsh 4.15.2 验证的特征线网格策略的**可选验收补丁**，不改变默认密度、不改变流动方程、不声称任何气动精度提升。

## 实测动机

椭圆翼同一 CAD、size=0.100 的真实本地 PR #43、Issue #42 报告表明：
- 不带特征的 1028 wall / 577 wake，最小角 10.22°；
- 只加前缘：约 1730 wall / 577 wake，曾出现最小角 6.44°；
- 前后缘共同加密：约 2500 wall / 1003 wake，尾迹也增长；
- target_ratio=0.125 的聚集：4924 wall，最小角 6.22°，物理载荷不单调。

因此几何尺寸回调的数学性质不能替代**实际单元质量**和内存预算。项目当前没有证据达到严格网格无关；局部加密更不能仅凭 CL 靠近其他档或更小 Cp 峰值宣布更准。

## 可选命令

    python -m panel.cad_half --wing-cad ORIGINAL.step --trailing-edge REAL_TE.json \
       --mesh-features VERIFIED_LE_TE_POLYLINES.json --size 0.10 --far-size 1.5 \
       --radius 5 --alpha-deg 5 --max-wall-panels 2200 --max-wake-panels 900 \
       --max-total-panels 3000 --min-wall-angle-deg 8 \
       --max-wall-growth-p99 3.5 --out runs/NEW_MESH_GATED

门禁默认均关闭；**未使用 --mesh-features 且未指定任何新门禁**时，不对输出增加字段。指定任意门禁或特征线时，在 OCC mesh.generate(2) 后检查实际壁面/尾迹计数和三角形质量，再继续原 build_half_surface 与 prepare_half_topology，不修改节点/拓扑。

- --max-wall-panels、--max-wake-panels、--max-total-panels：真实生成的 2D 三角形计数上限，拒绝 ValueError，不会静默增大尺寸。此门禁位于 Gmsh 剖分之后，因此阻止超预算进入后续稠密求解，**不保证 mesher 本身的内存安全**。
- --min-wall-angle-deg：真实湿壁三角形最小内角。例中设为 8° 将拒绝出现 6.44° 的 sliver；8° 仅为可复现实验示例，**不是普适精度标准**。
- --max-wall-growth-p99：共享真实壁面边两侧三角形面积比的 p99，不包含尾迹，不把自适应变化误当错误；阈值由实际分布决定，不能凭空优化。
- mesh_quality 输出：wall/wake 三角形计数、最小角、p01角、三角形质量、边长比p99、实际壁面邻接面积比p99与最差三角索引。网格本身和所有旧报告只读。
- provenance.feature_sizing.requested_features 记录每条特征线 target_ratio、radius_chords、源折线段数，并保留原输入 SHA；不必额外打开原 JSON 才能知道所请求的加密程度。

## 本轮真实验证状态与下一步

这套 panel.mesh_quality_gate 纯 NumPy 核心已经实际执行 pytest -q test/test_mesh_quality_gate.py：**17 passed**；它覆盖几何、非流形边、角度/增长拒绝、计数预算、非法参数、输入不改。当前云端**没有安装 Gmsh**，因此新的 cad_half.py CLI 接线、实际提交文件的旧无flag回归及真网格拒绝路径仍需本地重新运行。

本地 agent 应在独立 worktree 验证：
1. 旧无 flag 椭圆翼 size 0.100；无门禁时输出新增字段为零；在相同本机环境下不要求网格坐标/SHA 位相等（此前已证实 Gmsh 并非逐位确定）。
2. 对原 feat_le 和 feat_lete 新目录分别运行宽松预算，然后以 --min-wall-angle-deg 8、--max-wall-panels 2200、--max-wake-panels 900 各自测试预期拒绝，并核对 failure.json 首异常和未覆盖旧目录。
3. 正式 A/B 的面元密度、CL、Cp、CM 和总流尾迹/物理误差仍需要相同模型和独立外部基准；跨工况采用多种重复网格种子或线程设置来估计网格生成不确定性，不把相邻 CL 差小于 5% 当 GCI。

与 #41 多涡片/PR #45 的真实 CAD 实现**完全独立**。physical_accuracy_accepted=false，不合并、不改默认、不启动 GRASS 训练。
