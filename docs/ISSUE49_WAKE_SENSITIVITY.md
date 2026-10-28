# issue #49 优先级 c：VSPAERO 尾迹设置（NumWakeNodes / WakeIters / WakeRelax）敏感性

`physical_accuracy_accepted = false`。本文只记录外部求解器 VSPAERO 3.55 在**固定网格**上对尾迹设置的敏感性，
目的是给 issue #49 的「DPanel vs VSP 厚 panel」比较标一个外部不确定性上界，而不是给 DPanel 做标定。

## 设置

- 网格与几何：`runs/vsp_conv/api_355_panel`（OpenVSP 3.55 厚面元，12996 个面元，梯形翼 trap，Sref 0.30 / Cref 0.15 / Bref 2.0）。
  所有扫描复用同一份 `trap.vspgeom` + `trap.vkey` 字节副本，只重写 `trap.vspaero` 的尾迹字段。
- 求解器：`D:/3D/Projects/OpenVSP-3.55.0-win64/vspaero.exe -omp 32 trap`（`NCPU` 字段无效，只有命令行 `-omp` 生效）。
- 迎角 5 度，其余（Vinf=100、Rho=0.002377、ReCref=1e7、Mach=0）逐字节沿用模板。
- 模板是 GUI 风格设置：`NumWakeNodes = 8`、`WakeIters = 5`、`WakeRelax = 1`、`FarDistToggle = 1`、`FarDist = -1`、`FreezeWakeAtIteration = 10000`。
- 脚本：`runs/vsp_conv/wake_sweep.py`（主扫描）、`runs/vsp_conv/wake_ref.py`（参照与远场档）、`runs/vsp_conv/wake_report.py`（生成本文）。

**复现检查**：把模板字段原样回填（nodes 8 / iters 5 / relax 1 / FarDist -1）重跑，得 CLtot = 0.709296、CDiw = 0.010943；
证据包 `docs/issue49_vspaero/ladders/api_355_panel/trap.polar` 对应值是 0.709350 / 0.010723：CLtot 差 -5.38e-05（-0.008%），CDiw 差 2.04%。
**这不是位级可重复**：vspaero 的 `-omp 32` 求和顺序会动末位，尾迹松弛本身也不是严格确定性的。所以本扫描的复现容差按 CLtot 0.01%、CDiw 2% 计。

## 1. NumWakeNodes（iters = 20，relax = 0.5）

| 档位 | CLtot | CLwtot | CDi | CDiw | CDwtot | CMytot |
| --- | --- | --- | --- | --- | --- | --- |
| nodes04 | 0.709157 | 0.711752 | 0.016935 | 0.010956 | 0.024467 | -0.18385 |
| nodes08 | 0.709520 | 0.712652 | 0.016795 | 0.010933 | 0.024452 | -0.18379 |
| nodes16 | 0.711125 | 0.714675 | 0.017486 | 0.010666 | 0.024200 | -0.18415 |
| nodes32 | 0.714104 | 0.717737 | 0.017621 | 0.010249 | 0.023807 | -0.18489 |
| nodes64 | 0.719341 | 0.722627 | 0.016788 | 0.009492 | 0.023089 | -0.18644 |

- CLtot 从 0.709157（nodes 4）单调升到 0.719341（nodes 64），跨度 +1.44%；nodes 8 到 32 已有 +0.65%。
- 尾迹涡量阻力 CDiw 反向单调降：0.010956 到 0.009492（-13.36%）；近场 CDi 在同一扫描里先升后降（0.016935 / 0.017486 / 0.016788），
  说明尾迹节点数主要在 CDi 与 CDiw 之间搬动量，总量 CDwtot 只从 0.02447 变到 0.02309（-5.63%）。
- 结论：**NumWakeNodes 是这台求解器上的一阶旋钮**，1.4% 的 CL 与 13% 的 CDiw 变化不需要碰网格。

## 2. WakeIters（nodes = 32，relax = 0.5）

| 档位 | CLtot | CDi | CDiw | CDwtot | CMytot |
| --- | --- | --- | --- | --- | --- |
| iters01 | 0.729126 | -0.008283 | 0.010252 | 0.023910 | -0.18173 |
| iters03 | 0.714704 | 0.017586 | 0.010262 | 0.023824 | -0.18507 |
| iters05 | 0.714363 | 0.017645 | 0.010258 | 0.023817 | -0.18497 |
| iters10 | 0.714195 | 0.017631 | 0.010240 | 0.023798 | -0.18492 |
| iters20 | 0.714104 | 0.017621 | 0.010249 | 0.023807 | -0.18489 |
| iters40 | 0.713917 | 0.017628 | 0.010260 | 0.023816 | -0.18484 |

- `WakeIters = 1` 没收敛：CLtot 0.729126、CDi **-0.008283（负值）**，不可用。
- 3 次之后进入平台：3 / 5 / 10 / 20 / 40 的 CLtot 为 0.714704 / 0.714363 / 0.714195 / 0.714104 / 0.713917，3 到 40 只差 -0.11%。
- 结论：这台网格上 5 次松弛迭代已足够，**迭代数不是误差源**；但 1 次会给出符号错误的 CDi，外部对比必须记录迭代数。

## 3. WakeRelax 与冻结尾迹（nodes = 32，iters = 20）

| 档位 | CLtot | CDi | CDiw | CDwtot | CMytot |
| --- | --- | --- | --- | --- | --- |
| relax02 | 0.714250 | 0.017660 | 0.010239 | 0.023797 | -0.18494 |
| relax05 | 0.714104 | 0.017621 | 0.010249 | 0.023807 | -0.18489 |
| relax10 | 0.713906 | 0.017621 | 0.010260 | 0.023817 | -0.18483 |
| frozen | 0.714681 | 0.017655 | 0.010232 | 0.023794 | -0.18507 |

- 松弛因子 0.2 / 0.5 / 1.0 的 CLtot 为 0.714250 / 0.714104 / 0.713906，跨度只有 0.05%：**在 nodes 32 上 relax 不是敏感项**。
- 需要修正早先的记录：`docs/VSPAERO_PANEL_POLAR_DIAGNOSIS.md` 的「relax = 1 会发散」是在**粗网格 + nodes 8** 上观察的；在 12996 面、nodes 32 上
  relax = 1 给出 0.713906（比 relax = 0.5 低 0.03%），并未发散。该结论不能外推。
- 冻结尾迹（`FreezeWakeAtIteration = 1`）给 0.714681，比迭代尾迹高 +0.08%。

## 4. FarDist（nodes = 32，iters = 20，relax = 0.5）

| 档位 | CLtot | CDi | CDiw | CDwtot | CMytot |
| --- | --- | --- | --- | --- | --- |
| FarDist = -1（自动） | 0.714104 | 0.017621 | 0.010249 | 0.023807 | -0.18489 |
| FarDist = 5 | 0.711112 | 0.017712 | 0.010623 | 0.024157 | -0.18415 |

- 只改远场距离，CLtot 变 -0.42%、CDiw 变 +3.65%：尾迹被截断的长度直接改变下洗。
- 所以 `FarDist` 必须和 `NumWakeNodes` 一起记录，否则同一个「VSP 厚 panel」标签下会有两个不同基准。

## 5. 对 issue #49 的含义（关键）

1. 证据包里的「VSP 厚 panel 12996 面 CLtot 0.704593」来自 `ladders/api_panel_74x170`（nodes 8 / iters 5 / relax 1，另一份 3.55 网格）；本次扫描的基网格
   `api_355_panel` 用同一套设置给 0.709350，两者差 0.67% 来自网格/几何集差异，**不能混用**。
2. 同一份网格只动 NumWakeNodes，CLtot 就从 0.709157 走到 0.719341（+1.44%）——这个跨度**大于**上一轮 Gamma 细化后剩下的 0.53% 差距。
3. 因此在尾迹离散未对齐前，「DPanel 5 度 CL 比 VSP 低 0.5%」这类说法不成立：残差落在 VSP 自身的尾迹设置不确定性里。
   要把这条差距做实，必须先固定 VSP wake 配置并用 `trap.lod` 复核尾迹形状，或者把 DPanel 尾迹离散做成与之匹配的一族。
4. 更要紧的是方向性：归档基准用的是 **nodes 8**，而 nodes 8 处在 CLtot 单调上升段的低端。换言之，「DPanel 5 度 CL 差 0.53%」是在最有利于 DPanel 的尾迹配置下取得的；
   换成 nodes 32 / 64，同一份 VSP 网格的 CLtot 升到 0.714104 / 0.719341，DPanel 反而低 1.9% / 2.6%。任何「已收敛到 1% 以内」的说法都必须绑定 wake 配置。
5. 对 DPanel 侧的可执行建议：外部对比把「VSP wake 配置」当成基准定义的一部分写进证据表，否则 1% 量级的 CL 与 6% 量级的 CDiw 差异无法归因。

## 6. 归档

- 原始读数：`runs/vsp_conv/wakesweep/results.json`、`results_ref.json`；每档一个目录（`trap.vspaero`、`trap.polar`、`stdout.log`）
- 脚本：`tools/vsp_evidence/wake/{wake_sweep.py,wake_ref.py,wake_report.py}`
- 文档：本文件由 `wake_report.py` 从 JSON 生成（只读，不重解）
