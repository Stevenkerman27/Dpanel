# 后缘：闭包（pressure / potential_jump）与速度恢复（face / area）

更新 2026-11-21。这一篇只回答三件事：两种后缘闭包在方程上差在哪、两种速度恢复在几何上
差在哪、遇到具体问题时该选哪一组。数字全部可用 §7 的命令在 `runs/verification/route_check/`
复算，默认链的其余部分见 [ROUTE.md](ROUTE.md)。

## 1. 结论（先看这段）

1. **默认是 `pressure` + `face`**（`panel/p1_lifting_g2.py:330` `PRODUCTION_KUTTA='pressure'`，
   `solve_multi` 的 `pressure_recovery="face"`）。浏览器工作台只跑这一组。
2. **`area` 恢复在测过的每一张网格上都不差于 `face`**：Galerkin 壁面下，把恢复从 `face` 换成
   `area` 后 **pressure 与 potential_jump 的结果重合到 1e-5 量级**（后掠 30° 的 CL
   0.299383 vs 0.299370、CDi 4.979e-3 vs 4.980e-3、环量锯齿 0.251 vs 0.250；矩形 0.320731 vs
   0.320719、e 0.9710 vs 0.9709）。`face` 下这一对在同样网格上差 15%（后掠）。
3. **“potential_jump 在后掠翼上锯齿、CDi 高 6 倍”这条历史结论要加两个限定**：
   其一，它是在 **collocation 壁面 + face 恢复**下测的（`tools/wake_closure_study.py` 的
   `run()` 第 129/152/189 行都写死 `wall_formulation="collocation"`）；其二，**face 恢复配
   pj 本身就是坏的**——展向加密时 e 从 0.377 掉到 0.099，再掉到 0.0028（CDi 16.4，pressure
   的 3000 倍、Γ 变号 56 次），而同一批网格换 `area` 就稳定在 e≈0.895–0.899、0 次变号。
   所以真正不能用的是“face + potential_jump”这一对（系统性不一致，随展向加密放大），
   不是 pj 方程；在 Galerkin + area 下它与 pressure 重合到 1e-5。

由此的工程建议（尚未改代码，属于决策项）：

- **把 `pressure_recovery` 默认改成 `area`**（`solve_multi` / `solve_multi_torch` /
  `fixed_wake_geometry` 三处），并用冻结算例复核一次梯度一致性；
- **闭包默认保持 `pressure`** 到 pj+area 有自己的门禁为止；门禁应包含“Γ 锯齿/最大展向斜率”
  这一条，而不只是 `Δp_TE` 残差——`Δp` 到了 9.3e-15 仍然可以是锯齿解（§5）。
- 旧口径的两处 docstring 过期（`p1_cst_gradient.py:555` 说生产闭包是 potential_jump、
  `p1_te_recovery.py:10` 说 area 才是生产选项），与代码默认相反，应在改默认时一并修。

## 2. 两种闭包差在方程结构

壁面偶极 μ 与尾迹环量 Γ 的离散方程：

    A μ + W Γ = b            (壁面：Galerkin 弱形式或配点)
    C μ − I Γ = 0            (后缘势跳变：上/下壁面的 μ 差 = 该站位的 Γ)

前两行只说明“Γ 是势跳变”，**没有决定 Γ 的大小**；闭包就是把 Γ 定下来的那条附加条件。
两种实现：

| | pressure（默认） | potential_jump |
|---|---|---|
| 附加条件 | 后缘上下表面的 ΔCp → 0（离散成罚项，`kut_weight=5`） | 同样要求 Δp_TE = 0，但作为**方程**精确求解 |
| 线性代数 | 超定最小二乘，`null(J)` 降维后对 μ 求最小；结构化 Gauss-Newton + B 预分解 + Woodbury（`p1_pressure_kutta_torch_driver.py:166`） | 方阵：先解 μ 对 Γ 的**仿射**关系（一次 lstsq），再对 Γ 做 Newton（`p1_potential_jump_torch.py`） |
| 鲁棒性 | Δp 只是权重 5 的罚项，Γ 由整片壁面 B 安放，不吃“整条展向 Γ 让每站 Δp 几乎不变”的近零方向 | 方阵 + Newton，存在多个根；粗网格上不同初值收敛到不同根（`KUTTA_CONDITION.md` §3.3） |
| 验收判据 | `pressure_kutta_acceptance`（`p1_lifting_g2.py:269`）：相对梯度 ≤1e-6、壁面残差 ≤1.05×floor、压力残差 ≤1e-4 | 无专用门禁，只看 Newton 是否收敛 |
| 自动回退 | — | 只有未 pinned 且抛 `"potential-jump Kutta..."` 时回退到 pressure，输出记 `kutta_requested/effective/fallback`，从不静默 |

物理上两者都要求后缘压力连续；差别只在“**怎么把这条要求塞进离散方程**”。所以它们的差
必然与壁面离散、恢复位置耦合——这就是 §4 那张表要分开写壁面和恢复的原因。

## 3. 两种恢复差在“在哪个位置比较上下表面”

后缘压力是“同一位置上下表面的速度差”算出来的，但离散里“上下表面的速度”有不止一种取法：

| 恢复 | 速度取自 | 位置是否一致 | 代码 |
|---|---|---|---|
| `face`（默认） | 每个岸上**声明的后缘面**的 P1 面梯度，再对每岸的若干面做算术平均 | **否**：上下岸用的是不同的面，面心位置不同；对后掠/有厚度的后缘，这些面心本身就相差一个弦向/法向偏移 | `p1_pressure_kutta_torch_driver.TorchPressureModel`（`mean_top|v|² − mean_bottom|v|²`）；配点路径另用 `p1_lifting_g2.py:253` 的“每站最近两个同岸面” |
| `area` | 在**同一个后缘节点**上，用面积加权的 one-ring 面梯度重建上下岸各自的速度矢量 | **是**：同一点、两个方向 | `p1_te_recovery.matched_te_velocity_maps(method="area")`（`p1_te_recovery.py:10`） |
| `quadratic` | 同一点，但用 1,u,v,u²,uv,v² 六项 QR 拟合，秩不足时扩环到 5 环 | 是 | 同上，`method="quadratic"`；研究控制，粗网格上 e 会掉到 0.04，**不是**生产选项 |

一句话：`face` 比较的是“两个不同位置的压力”，`area` 比较的是“同一位置的两个压力”。
对有限夹角、有厚度的后缘，这两种离散不等价；旧文档里“各岸邻面速度平方作算术平均”说的
就是 `face`。

## 4. 同一组网格上的 A/B（本轮实测）

Galerkin 壁面、固定尾迹、α=4°、AR=6、单线程。列：CL / Trefftz CDi /
e = CL_γ²/(π·AR·CDi) / Σ|ΔΓ|（环量锯齿度量）/ Γ 变号次数。

后掠 30°（1560 三角形，782 节点）：

| 壁面 | 恢复 | 闭包 | CL | CDi | e | Σ\|ΔΓ\| | 变号 |
|---|---|---|---:|---:|---:|---:|---:|
| collocation | face | pressure | 0.278962 | 4.218e-3 | 0.898 | 0.454 | 0 |
| collocation | face | potential_jump | 0.345508 | 3.984e-2 | 0.147 | 6.972 | 4 |
| galerkin | face | pressure | 0.298891 | 4.936e-3 | 0.903 | 0.367 | 0 |
| galerkin | face | potential_jump | 0.343703 | 6.017e-2 | 0.099 | 8.762 | 6 |
| galerkin | **area** | pressure | 0.299383 | 4.979e-3 | 0.898 | 0.251 | 0 |
| galerkin | **area** | potential_jump | 0.299370 | 4.980e-3 | 0.898 | 0.250 | 0 |

矩形（782 节点）：

| 壁面 | 恢复 | 闭包 | CL | CDi | e | Σ\|ΔΓ\| | 变号 |
|---|---|---|---:|---:|---:|---:|---:|
| galerkin | face | pressure | 0.319943 | 5.369e-3 | 0.945 | 0.887 | 0 |
| galerkin | face | potential_jump | 0.321231 | 6.324e-3 | 0.809 | 2.214 | 0 |
| galerkin | **area** | pressure | 0.320731 | 5.253e-3 | 0.971 | 0.311 | 0 |
| galerkin | **area** | potential_jump | 0.320719 | 5.253e-3 | 0.971 | 0.310 | 0 |

椭圆（782 节点）：

| 壁面 | 恢复 | 闭包 | CL | CDi | e | Σ\|ΔΓ\| | 变号 |
|---|---|---|---:|---:|---:|---:|---:|
| collocation | face | pressure | 0.305888 | 5.038e-3 | 0.907 | 0.599 | 2 |
| collocation | face | potential_jump | 0.332800 | 5.460e-3 | 0.992 | 0.524 | 0 |
| collocation | area | pressure | 0.306466 | 5.003e-3 | 0.918 | 0.561 | 2 |
| collocation | area | potential_jump | 0.332932 | 5.463e-3 | 0.992 | 0.523 | 0 |
| galerkin | face | pressure | 0.322037 | 5.229e-3 | 0.992 | 0.515 | 0 |
| galerkin | face | potential_jump | 0.321988 | 5.228e-3 | 0.991 | 0.515 | 0 |
| galerkin | **area** | pressure | 0.322166 | 5.232e-3 | 0.992 | 0.512 | 0 |
| galerkin | **area** | potential_jump | 0.322121 | 5.230e-3 | 0.992 | 0.512 | 0 |

读法：

- **Galerkin + area**：三个平面形状上两闭包的 CL 差 ≤1.3e-5、CDi 差 ≤1e-6。
- **Galerkin + face**：椭圆无差别（0.992），矩形 e 从 0.971 掉到 0.809，后掠直接崩。
- **collocation**：pressure 在椭圆上 e 只有 0.907（“CL 低 7%”的来源），而 pj 是 0.992；
  但 collocation 的 pressure 解即使换 area 也回不来（0.918），说明配点壁面本身是瓶颈，
  与恢复无关。
- 换 `area` 对 pressure 也普遍略好（矩形 e 0.945 → 0.971，后掠 Σ|ΔΓ| 0.367 → 0.251）。

## 5. 那个 6 倍阻力到底是怎么回事

同后掠 30° 平面形状，只改弦向分段（Galerkin，固定尾迹）：

| 弦向分段 | 壁面节点 | face + pressure (CL / e / Σ\|ΔΓ\|) | face + pj | area + pressure | area + pj |
|---|---:|---|---|---|---|
| 4 | 522 | 0.299623 / 0.879 / 0.678 | 0.300518 / **0.575** / 3.444 | 0.300505 / 0.891 / 0.241 | 0.300445 / 0.890 / 0.240 |
| 6（默认） | 782 | 0.298891 / 0.903 / 0.367 | 0.343703 / **0.099** / 8.762 | 0.299383 / 0.898 / 0.251 | 0.299370 / 0.898 / 0.250 |
| 8 | 1042 | 0.296543 / 0.905 / 0.267 | 0.296535 / 0.905 / 0.267 | 0.296831 / 0.900 / 0.255 | 0.296829 / 0.900 / 0.255 |

弦向细化下 `face + potential_jump` 的 e 是 0.575 → 0.099 → 0.905（非单调），看上去像
偶发的坏根。再把**展向**分段从每段 3 等分换到 5 等分（都固定弦向 6 分段）：

| 展向每段分段 | 壁面节点 | face + pressure (CL / e / Σ\|ΔΓ\|) | face + pj | area + pj |
|---|---:|---|---|---|
| 3 | 590 | 0.300017 / 0.871 / 0.710 | 0.318495 / **0.377** / 3.403 | 0.298829 / 0.895 / 0.224 |
| 4（默认） | 782 | 0.298891 / 0.903 / 0.367 | 0.343703 / **0.099** / 8.762 | 0.299370 / 0.898 / 0.250 |
| 5 | 974 | 0.299281 / 0.903 / 0.277 | 0.198734 / **0.0028** / 252.2（56 次变号！） | 0.299704 / 0.899 / 0.269 |

展向越密越糟：e 0.377 → 0.099 → 0.0028，CDi 从 1.3e-2 涨到 **16.4**（pressure 是 5e-3，
差 3000 倍），Γ 有 56 次变号、最大展向斜率 5.6。同一张网格上的 `area + pj` 始终
0.895–0.899、0 次变号。所以这**不是**偶发坏根，而是 `face` 恢复与 `potential_jump` 方程
之间的系统性不一致，随展向加密放大。

逐项查 c6/span4 那个坏解：

    CL 0.343703, CDi 6.017e-2, e 0.098, Σ|ΔΓ| 8.762, 6 次变号
    pressure_kutta_rms 9.27e-15   ← Δp_TE 已经被压到机器精度
    wall_rms 4.15e-6              ← 壁面也解开了
    signed_kutta_rms 0.0508       ← 但后缘势跳变/环量的一致性很差（span5 时涨到 1.35）

它是一个**收敛得很好的坏解**：后缘压力连续、壁面满足、但环量锯齿。`Δp_TE` 残差型门禁
抓不到它——这也是当初"条件数门禁"没报警的原因。

同时，`face + pressure` 在这些网格上一直正常（e 0.879–0.905），所以问题不在"后掠翼"
本身，而在 **face 恢复 + pj 这一对**。

椭圆翼的弦向细化（Galerkin，CL / e）：

| 弦向分段 | area + pressure | area + pj |
|---|---|---|
| 4（522 节点） | 0.324650 / 0.9935 | 0.324579 / 0.9935 |
| 6（782 节点） | 0.322166 / 0.9918 | 0.322121 / 0.9918 |
| 8（1042 节点） | 0.318697 / 0.9915 | 0.318663 / 0.9914 |

两闭包在每一档都重合（CL 差 ≤8e-5、e 差 ≤4e-4），e 稳定在 0.991–0.994。但 CL 随加密
0.3247 → 0.3222 → 0.3187 在**下降**、并没有趋向升力线的 0.328987；这个趋势与闭包无关，
更像固定尾迹/远场下的离散偏差，属于下一层要查的问题，不要用这张表推断"收敛到真值"。

**教训**：判断"某个闭包在后掠翼上不能用"，必须先固定壁面离散与恢复方式，否则测到的是
`（闭包 × 恢复）` 的组合效应。2026-11-10 那次默认值翻转（改 pj 又改回）用的是
**collocation + face**；用生产口径（Galerkin + area）复测，这个结论不成立。


## 6. 怎么选（可执行）

1. 默认（Web、CLI、优化）：`wall_formulation="galerkin"`、`kutta=None`（= pressure）。
2. **要做后缘/环量对比实验**：加 `pressure_recovery="area"`，否则你测到的是 `face` 的
   位置误差，而不是闭包差异。
3. 只关心升力/阻力趋势、追成本：`face` 也能用；但别用它的 `Σ|ΔΓ|` 或后掠翼的 CDi 做结论。
4. 想用 `potential_jump`：只配 `area`，并且人工看一眼 Γ 的展向变号次数；
   配 `face` 的解可能是锯齿伪根。
5. `quadratic`：研究对照，不要当生产选项（粗网格 e 会掉到 0.04）。
6. `collocation` 壁面：除非你在复现历史结论，否则不要用来评价闭包（§4 椭圆行）。

## 7. 复算

```bash
# 三平面形状 × 壁面 × 恢复 × 闭包（§4）
python runs/verification/route_check/run.py          # swept30/elliptic → swept_elliptic.json
python runs/verification/route_check/rect.py         # rect + elliptic collocation → rect_colloc.json
# 弦向细化（§5）
python runs/verification/route_check/refine.py       # → refine.json
# 展向细化 + 坏解诊断（§5）
python runs/verification/route_check/root_probe.py        # → root_probe.json
python runs/verification/route_check/span_face_pressure.py # → span_face_pressure.json
```

单线程（`torch.set_num_threads(1)`），每个解 2–30 s；12 个解约 3 分钟。JSON 里偶有 `NaN`
（`span_e`），解析前替换成 `null`。

**不声明**：以上都是单网格族、α=4°、固定尾迹、无粘；`physical_accuracy_accepted` 仍为
`false`。这些数字支持“离散选择”层面的结论，不支持“气动精度已验收”。
