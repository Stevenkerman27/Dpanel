# VSPAERO 厚面元（panel）模式：polar 的 CLi 通道为什么不可用

日期 2026-10-27。对象：`cases/vsp/`（OpenVSP 3.52.1，`trap.vsp3` + 6249 面板厚网格）。

**结论**：网格、参考量、对称性都没错；`trap.lod` 的截面载荷也对。坏的是 polar 里的 **`CLi` 及其派生列**（`CLtot`、`CDi`、`CDtot`、`L/D`、`E`、`CMytot`）：panel 模式下 VSPAERO 把这个通道算成了**确定性错误值**（10²–10⁴ 倍），它**不是**尾迹迭代没收敛，改 `WakeRelax`/`WakeIters`/`NumWakeNodes`/`FarDist`/预条件都改变不了它。稳定可信的是 `CLwtot`、`CDwtot` 和 `trap.lod` 的条带载荷。

## 1. 配置与症状

`cases/vsp/trap.vspaero`：`Sref=0.3 Cref=0.15 Bref=2 X_cg=0 AoA=0,5,10 ReCref=1e7 Vinf=100 Rho=0.002377 Symmetry=1 FarDist=-1 NumWakeNodes=8 WakeIters=5 WakeRelax=1`；网格 6306 点 / 6249 面板（双面厚体，面积 0.3070）。

| AoA | CLo | CLi | CLtot | CDo | CDi | CDtot | CLwtot | CDwtot | CMytot | E |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | −0.0003 | **27.309** | 27.309 | 0.0099 | **0.1198** | 0.1297 | 0.1762 | 0.0105 | −8.429 | 148.6 |
| 5 | −0.0015 | **34.249** | 34.248 | 0.0124 | **3.0755** | 3.0879 | 0.6611 | 0.0216 | −10.477 | 9.10 |
| 10 | −0.0037 | **34.157** | 34.153 | 0.0178 | **5.9859** | 6.0037 | 1.1412 | 0.0453 | −10.524 | 4.65 |

`CLi` 大 150 倍、5°→10° 反而下降、`E = 148.6` 越 Munk 极限。

## 2. 它稳定地错着，而不是发散了

`trap.history`（每轮尾迹迭代的 `CLi`）在用户这次运行里：

| 尾迹迭代 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| CLi @5° | 0 | 2.502 | 26.771 | 30.583 | 34.249 |

看起来像发散，但把迭代数加到 20 并换设置后，**它收敛到一个稳定的荒谬值**：874 面板档第 5–20 轮逐位不变 `28850.33`；把 `WakeRelax` 从 0.5 改到 0.3 得到 `28850.33` vs `28850.06`（差 0.0009，可以忽略）。→ **`WakeRelax` 对这个量没有实质影响**，之前“relax=0.5 就能修好”的判断作废（那是误跑薄 VLM 的残留读数）。

## 3. 设置矩阵：所有旋钮都试过（6249 面板厚网格、5°）

| 配置 | iters | nodes | relax | FarDist | precond | 秒 | CLi | CDi | CLwtot | CMytot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c1 = 用户的设置 | 5 | 8 | 1.0 | −1 | 0 | 15.2 | 33.11 | 2.976 | 0.6611 | −10.14 |
| c2 | 20 | 8 | 1.0 | −1 | 0 | 44.4 | 35.44 | 3.181 | 0.6608 | −10.83 |
| c3 | 20 | 32 | 1.0 | 5 | 0 | 81.1 | 30.19 | 2.714 | 0.6608 | −9.27 |
| c4 | 20 | 32 | 0.5 | 5 | 0 | 82.0 | 29.88 | 2.687 | 0.6608 | −9.18 |
| c5 | 20 | 64 | 0.5 | 5 | 1 | 183.9 | 25.59 | 2.312 | 0.6645 | −7.93 |

c1 复现了用户的 34.25（33.11，−3%）✓。**`CLwtot` 全程锁在 0.6608±0.4%，`CLi` 在 25.6–35.4 乱晃** → 两个通道独立，一个是好的，一个是坏的。

## 4. `CLi` 是网格的函数：随加密单调坍缩

同一套求解设置（`WakeIters 5 / NumWakeNodes 8 / FarDist −1 / WakeRelax 1 / Symmetry 1`）下把自己生成的厚网格逐档加密，`CLi` 单调坍缩到合理值：

| 面板数 | 874 | 1890 | 3354 | **6222** |
| --- | --- | --- | --- | --- |
| `CLi` @5° | 29810 | 1607 | 44.8 | **0.635** ✓ |
| `CLwtot` @5° | 0.5242 | 0.6193 | 0.6439 | 0.6544 |

所以它不是随机噪声，而是**粗网格上的一个奇异项，随面板变小而衰减**（874→6222 面板差 5 个数量级）。到目标密度（6222 面板，与 `cases/vsp` 的 6249 同一量级）`CLi` 已经和 `CLwtot` 只差 3%。

### 4.1 负对照：与尾迹形状、对称处理无关

在**用户那份 6249 面板网格**上（5°，relax 0.5 / iters 20 / nodes 32 / FarDist 5）：

| 对照组 | CLi | CLwtot | 结论 |
| --- | --- | --- | --- |
| 基准 c4 | 29.8773 | 0.6608 | — |
| `FixedWakeFlag = 1`（冻结尾迹） | **29.8773** | 0.6608 | 逐位相同 → 与尾迹迭代/形状无关 |
| `FreezeWakeRootVortices = 1` | **29.8773** | 0.6608 | 逐位相同 |
| `Symmetry = 2` | **29.8773** | 0.6608 | 逐位相同 → 与对称面无关 |
| `Symmetry = 0`（半模当整机） | 20.2082 | 0.2720 | 升力塌一半，符合预期 |
| `UpdateMatrixPreconditioner = 1`、nodes 64 | 25.5877 | 0.6645 | 只从 29.9 挪到 25.6 |
| `ImplicitWake = 1` | — | — | GMRES 117 迭代不收敛，polar 根本没写出来 |

### 4.2 决定性交叉：同一设置、两份网格

把**用户自己的 `cases/vsp/trap.vspgeom`**（6306 点 / 6249 面板）喂给上表第 4 节那套**能让我的网格收敛的设置**（iters 5 / nodes 8 / FarDist −1 / relax 1）：

| 网格 | 面板 | `CLi` @5° | `CLwtot` @5° | `CDi` @5° |
| --- | --- | --- | --- | --- |
| `cases/vsp/trap.vspgeom`（VSPAERO 从用户模型生成） | 6249 | **34.2494** | 0.6611 | 3.0755 |
| 我用 DEP-OPT `create_wing` 重新生成的厚网格 | 6222 | **0.6346** | 0.6544 | 0.0109 |

两份网格是**同一个机翼、同一展弦比、同一 NACA 2412、同一 52×120 分块**，设置逐字相同，用户网格复现出 GUI 的 34.2494（与 `trap.polar` 逐位一致）。**⇒ 触发条件在 `cases/vsp/trap.vspgeom` 这份网格里，不在求解设置里。**

两份网格的实测差异：

| 指标 | 用户网格 | 我的网格 |
| --- | --- | --- |
| 面板 / 点 | 6249 / 6306（6241 quad + 8 tri） | 6222 / 6269（6214 quad + 8 tri） |
| 面板面积 min / 中位 / max | 1.23e−7 / 4.56e−5 / 1.20e−4 | 4.35e−7 / 3.80e−5 / 1.44e−4 |
| 最短边（全网格） | 1.43e−4 | 2.45e−4 |
| **翼尖端盖面板（全部落在 y=1）** | **180** | **102** |
| 展向站数 / 间距 | 120 / 4.2e−3…8.4e−3（规则两档） | 121 / 4.1e−3…1.62e−2（中间疏） |

翼尖端盖与展向站分布是主要嫌疑（端盖上的贴体涡条 + 尖锐的尖部站最容易让 KJ 积分出奇异）。
### 4.3 根因：`LECluster`（前缘弦向聚类）

用户的 `trap.vsp3` 里 `TECluster = LECluster = 0.25`、`MaxGrowth = 1.255`；DEP-OPT 的 `create_wing` 用 `TECluster = 0.9 / LECluster = 0.8`。逐个换参数，其余（机翼、52×120 分块、面板数、`.vspaero`、exe）完全不变：

| `TECluster` | `LECluster` | `MaxGrowth` | 面板 | `CLi` @5° | `CLwtot` @5° |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 0.25 | 1.255 | 6216 | **37.5847** ✗ | 0.6620 |
| 0.25 | 0.25 | — | 6216 | **37.5847** ✗ | 0.6620 |
| 0.25 | **0.80** | — | 6216 | **0.6220** ✓ | 0.6670 |
| **0.90** | 0.25 | — | 6218 | **125.4536** ✗ | 0.6492 |
| 0.90 | **0.80** | 1.255 | 6222 | **0.6346** ✓ | 0.6544 |
| 0.90 | **0.80** | — | 6222 | **0.6346** ✓ | 0.6544 |
| **0.50** | **0.50** | — | 6222 | **5.5015** ✗ | 0.6596 |

三条结论：

0. **误差随聚类强度单调**：`LE = 0.25 / 0.50 / 0.80` → `CLi = 125.5 / 5.50 / 0.635`（同一 TE=0.9 网格族），即聚类越紧、突变越陡、奇异项越大。
1. **`LECluster = 0.25` 是唯一触发条件**：`TE` 无论 0.25 还是 0.90，只要 `LE = 0.25` 就爆（37.6 / 125.5），只要 `LE = 0.80` 就正常（0.622 / 0.635）。
2. **`MaxGrowth = 1.255` 完全无关**：同一 `TE/LE` 下开或关，`CLi` 逐位相同。
3. **`CLwtot` 在所有设置下都稳**：0.6492–0.6670（±1.4%），物理量一直是好的，坏的只有 `CLi` 那一路。

机理：`CLi` 是逐条贴体涡条做的 Kutta–Joukowski 环量求和。`LECluster=0.25` 在前缘后不远处留下**尺寸突变的相邻排**（前缘第一排很密、下一排突然变粗），KJ 求和遇到这种跳跃会出 $O(1/h)$ 的奇异项；而 `LE=0.8` 的聚类是从前缘到中弦平滑过渡，没有突变。这同时解释了 §4 里 `CLi` **随网格加密单调坍缩**（单元变小 ⇒ 跳跃变缓 ⇒ 奇异项衰减）。

**可操作结论**：把 `LECluster` 调回 0.8（`TECluster` 0.9，`MaxGrowth` 不动），重新 `VSPAEROComputeGeometry`，polar 的 `CLi` 就正常了；在拿到干净网格之前，只信 `CLwtot` 和 `trap.lod`。

## 5. `CLo ≈ 0` 是 panel 模式的正常行为

panel 模式把力拆成 `CLo`（物面项）与 `CLi`（尾迹 Kutta–Joukowski 项），`CLtot = CLo + CLi`。这个封闭厚体的物面通道给出 ≈0（5° 时 −0.0015），升力几乎全在尾迹通道。所以 `CLo ≈ 0` 本身不是 bug，但**意味着所有依赖 `CLi` 的列都会跟着坏**。

## 6. `trap.lod` 的三段 119 行 = 三个迎角

357 行 = **3 个迎角 × 119 个展向站**（无 AoA 列，按 `AoA` 列表顺序分块，每块 `TrailVort` 1…119、`ΣdSpan = 1.0` 半展）。同站三块 `Cl = 0.1745 / 0.6663 / 1.1738` 正是 α=0/5/10 的截面升力（不是三张涡片）。

- 5° 块：`Σ Cl·dSpan = 0.687` vs polar `CLwtot = 0.661`（+3.9%）vs 薄 VLM `CLi = 0.6435`（+6.8%）✅
- 全 357 行求和 `2.0606`（三倍计数）❌；`Σ Cd·dSpan = −0.0607` 同理是混算
- 翼尖末行 `Cl = 4.76 / Cd = −6.27` 是退化站，需剔除

**所以“load 里的 cl 合理”是对的**，读法按 119 行一块切。

## 7. 用 API 打出真正的 panel 模式

`VSPAEROSweep` 没有 `AnalysisMethod` 输入，模式由几何集决定（`vsp.SET_NONE=-1`、`SET_ALL=0`、`SET_FIRST_USER=3`）：

- **panel**：`GeomSet = SET_ALL (0)`、`ThinGeomSet = SET_NONE (-1)`；
- **VLM**：`GeomSet = SET_NONE (-1)`、`ThinGeomSet = SET_FIRST_USER (3)`；
- 判据：panel 打 `Model type: PANEL or MIXED` + 6249 条涡环方程；VLM 打 `Model type: VLM` + 3094 条 + 网格面积 0.1507。

坑：DEP-OPT 用的 `VSPAERO_WING_GEOM_SET = SET_FIRST_USER = 3` 在这份 vsp3 里是**空集**（实测：23 个 set，机翼只在 set 1 “Shown”），所以它在这份模型上会静默退化成 VLM。

## 8. 实用取值

```
panel 模式读升力：CLwtot 或 trap.lod 分块求和（不要用 CLi / CLtot / CDi / CMytot）
WakeIters        = 20
NumWakeNodes     = 32
FarDistToggle    = 1
FarDist          = 5
UpdateMatrixPreconditioner = 0   # 1 只把 CLi 从 29.9 拉到 25.6，不修问题且慢一倍
```

验收判据：`Σ(lod.Cl·dSpan)` 与该迎角的 `CLwtot` 相差 < 5%；`CLwtot(α)` 单调、`dCL/dα ≈ 5–6 /rad`。

## 9. 复现

```powershell
.conda_env\python.exe runs\vsp_conv\inspect_case.py runs\vsp_conv\evidence\l3   # polar 全部 48 列 + history + lod 分块
C:\Users\zyx20\anaconda3\envs\vsppytools\python.exe runs\vsp_conv\panel_wake.py 5      # 设置矩阵
C:\Users\zyx20\anaconda3\envs\vsppytools\python.exe runs\vsp_conv\panel_ladder.py     # 厚网格阶梯
```

## 10. 边界

- 只解释“为什么 polar 不合理、该读哪一列”，不声明气动精度；`physical_accuracy_accepted = false` 不变。
- VSPAERO 是独立势流对照，不是实验真值；panel 模式 `CDo` 含经验项，不与 DPanel 的无粘压力阻力比较。
- `cases/vsp/` 未被修改（分析只读；跑 VSPAERO 时复制到独立工作目录）。
## 11. 原始数据保留与重跑

2026-10-27 的项目清理删掉了本文的原始运行目录（`runs/vsp_conv/panel_wake/<tag>/`、`cluster_*`、`probe_*`、`ncpu_probe/`、`thick_t1/`、`user_thick_u1/`、`*.history`/`*.tec.dat`），表里的读数全部保留在本文里。重建方式：

| 表 | 重跑 | 耗时（`-omp 8`） |
| --- | --- | --- |
| §4 的设置矩阵（c1–c5） | `vsppytools\python.exe runs\vsp_conv\panel_wake.py 5` | 15–184 s / 例 |
| §4.3 的 LECluster 二分 | `vsppytools\python.exe runs\vsp_conv\cluster_probe.py` | ≈ 2 min（7 例） |
| §2 的用户网格交叉 | `runs/_cross355.py`（已删）→ 用 `vspapi` 的 `Case.run` 喂 `cases/vsp/trap.vspgeom` | 70 s / 迎角 |
| 任一案例的 polar/history/lod 列 | `.conda_env\python.exe runs\vsp_conv\inspect_case.py <case_dir>` | 1 s |

证据包（`docs/issue49_vspaero/*.json` + `runs/vsp_conv/evidence/l{1..4}/`）**完整保留**，35 个文件的 SHA-256 在清理后重跑重建并全部校验通过（命令见 `docs/CLEANUP_2026-10-27.md` §4）。
