# issue #49 外部对照证据：VSPAERO 原始输出

本目录是 issue #49 回执 <https://github.com/Stevenkerman27/Dpanel/issues/49#issuecomment-6435519407> 要求的
「VSPAERO 原始输出归档」：气动系数、截面载荷、Cp 分布、收敛日志与配置文件都按源样保留，**没有为对齐 DPanel
改动过 `Sref`、`Γ`、`Cp` 或任何参考量**。

`sha256.json` 覆盖本目录全部原始与数值文件（仓库相对路径，README 自身除外），校验方法见第 5 节。

## 1. 版本与模式矩阵

| 表 | OpenVSP / 网格 | 求解器 | 模式 | GeomSet / ThinGeomSet |
| --- | --- | --- | --- | --- |
| `levels.json`（`raw/l1`–`raw/l4`） | 3.52.1 API（DEP-OPT `create_wing`） | 3.52.1 `vspaero.exe` | VLM 薄壳 | `-1 / 3` |
| `ladders/api_panel_*` | 3.55.0 API（`vspapi`） | 3.55.0 `vspaero.exe -omp N` | panel 厚面元 | `3 / -1` |
| `ladders/api_vlm_*` | 3.55.0 API（`vspapi`） | 3.55.0 `vspaero.exe -omp N` | VLM 薄壳 | `-1 / 3` |
| `dpanel/*.json` | DPanel `panel.cad_half`（Gmsh） | DPanel JAX 半模（commit `535f907`） | 厚面元（线性涡环 + 尾迹） | — |

**模式由几何集成员关系决定，不由 `SET_ALL` 决定**：`GeomSet = SET_FIRST_USER(3)` + `ThinGeomSet = SET_NONE(-1)`
剖出厚面元（湿面积/平面面积 ≈ 1.02），反过来剖出 VLM 薄片（≈ 0.50）。判据看 stdout 的 `Model type:` 行与
`Number Of Vortex Loop Equations`。

## 2. 收敛阶梯

所有表里 `CLtot` 是 VSPAERO 的面元表面积分升力，`CLwtot` 是尾迹诱导（Kutta–Joukowski）通道的升力；两者差
0.3%–3%，都在原始 polar 里（`ladders/*/trap.polar` 的 48 列）。

### 2.1 VSPAERO VLM，3.52 阶梯（`levels.json`，5° 为单点复跑）

| 档 | 面板 | CLtot(0°) | CLtot(5°) | CLwtot(5°) | CDi(5°) | CDtot(5°) | CMytot(5°) | 日志 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| l1 | 240 | 0.176702 | 0.641479 | 0.641916 | 0.010381 | 0.033163 | -0.1767 | `raw/l1/stdout.log` |
| l2 | 854 | 0.184471 | 0.643757 | 0.644199 | 0.011504 | 0.034311 | -0.1953 | `raw/l2/stdout.log` |
| l3 | 3120 | 0.188174 | 0.642537 | 0.643210 | 0.013955 | 0.036757 | -0.2057 | `raw/l3/stdout.log` |
| l4 | 12480 | 0.189350 | 0.640883 | 0.641633 | 0.015953 | 0.038743 | -0.2122 | `raw/l4/stdout.log` |

### 2.2 VSPAERO panel 厚面元，3.55 阶梯（同一 mesher，作主要对照）

| 档 | 面板 | 湿/平面面积 | CLtot(0°) | CLtot(5°) | CLwtot(5°) | CDi(5°) | CDtot(5°) | CMytot(5°) | lod ΣCl·dSpan | 秒 | `-omp` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 24x036 | 888 | 1.020 | 0.230296 | 0.718695 | 0.719679 | 0.013605 | 0.027184 | -0.1849 | 0.709499 | 9.2 | 8 |
| 34x052 | 1908 | 1.021 | 0.229551 | 0.719995 | 0.721183 | 0.013508 | 0.027102 | -0.1872 | 0.710314 | 31.0 | 8 |
| 44x076 | 3388 | 1.022 | 0.227668 | 0.716975 | 0.718759 | 0.015140 | 0.028714 | -0.1863 | 0.707128 | 83.2 | 8 |
| 52x120 | 6292 | 1.022 | 0.225267 | 0.709308 | 0.712561 | 0.016846 | 0.030364 | -0.1838 | 0.699356 | 44.4 | 8 |
| 74x170 | 12996 | 1.023 | 0.222424 | 0.704593 | 0.708639 | 0.018146 | 0.031629 | -0.1825 | 0.694761 | 111.5 | 16 |

### 2.3 VSPAERO VLM，3.55 阶梯（同一 mesher，作 2.2 的对照）

| 档 | 面板 | 湿/平面面积 | CLtot(0°) | CLtot(5°) | CLwtot(5°) | CDi(5°) | CMytot(5°) | 秒 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 24x036 | 420 | 0.501 | 0.181231 | 0.640673 | 0.641029 | 0.010297 | -0.1558 | 2.0 |
| 34x052 | 918 | 0.501 | 0.185680 | 0.644535 | 0.644677 | 0.010892 | -0.1600 | 5.2 |
| 44x076 | 1650 | 0.501 | 0.186797 | 0.643889 | 0.644037 | 0.011759 | -0.1609 | 12.3 |
| 52x120 | 3094 | 0.501 | 0.186629 | 0.640483 | 0.640831 | 0.013756 | -0.1605 | 27.3 |
| 74x170 | 6422 | 0.501 | 0.187771 | 0.639791 | 0.640177 | 0.014353 | -0.1617 | 17.8 |

### 2.4 DPanel，五档（同一几何）

| 档 | 壁面 | 尾迹 | γ dof | CL_p(0°) | CL_p(5°) | CL_Γ(5°) | CD_Trefftz(5°) | CM_y(5°) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dp_s020 | 1004 | 560 | 16 | 0.199972 | 0.630726 | 0.642821 | 0.009391 | -0.0633 |
| dp_s014 | 1822 | 642 | 16 | 0.196688 | 0.656163 | 0.653719 | 0.009637 | -0.0567 |
| dp_s010 | 3382 | 938 | 16 | 0.199323 | 0.684482 | 0.667294 | 0.010055 | -0.0533 |
| dp_s007 | 6562 | 1455 | 16 | 0.195510 | 0.694798 | 0.672085 | 0.010218 | -0.0501 |
| dp_s005 | 12290 | 2508 | 16 | 0.194801 | 0.695979 | 0.675226 | 0.010276 | -0.0483 |

DPanel 的 `CL_Γ` 用物理展向积分（PR #51 修正后的 `dy`，不是尾缘弧长 `ds`）；`CD_Trefftz` 已按真 `Sref = 0.30`
归一化（不是 `span × MAC = 0.3111`）。

## 3. 清单逐条对应（回执 6435519407）

| 要求 | 位置 |
| --- | --- |
| OpenVSP/VSPAERO 具体版本 | 第 1 节；`raw/lX/run.txt` 与 `ladders/*/stdout.log` 的 banner |
| 厚体 panel 还是薄 VLM、GeomSet/ThinGeomSet | 第 1 节；每档 stdout 的 `Model type:` 行 |
| CAD/TE/翼型/后缘厚度 SHA | `docs/VSP_TRAPEZOID_CONVERGENCE.md` §2；`cases/vsp/trap.vspgeom`、`runs/vsp_conv/cad_trap/{wing.step,trailing_edge.json}` |
| 网格与 wake 长/迭代/松弛 | 每档 `trap.vspaero`（`WakeIters`、`NumWakeNodes`、`WakeRelax`、`FarDist`）；`raw/lX/trap.vspgeom` |
| α/β/Mach/Re | `trap.vspaero`（`AoA`、`Beta`、`Mach`、`ReCref = 1e7`）与 polar 行 |
| Sref/bref/cref/参考点 | `Sref = 0.30`、`Bref = 2.0`、`Cref = 0.15`（3.52 档）/ `0.20`（3.55 档）；`X_cg = Y_cg = Z_cg = 0`；DPanel 的 `reference_chord` 是 MAC `0.15555556`，`moment_reference = [0.0388889, 0, 0]` |
| symmetry | 全部 `Symmetry = 1`（半模，y=0 对称面） |
| CL/CDi/CDtot/CM 原始值 | `levels.json`（VLM 3.52）、`ladders/*/ladder.json` 与 `ladders/*/trap.polar`（48 列全表） |
| 截面 Cp 或 ΔCp | `sections.json` 与 `raw/lX/trap.tec.dat`（`VARIABLES=x,y,z,Cp`，`VARLOCATION=([1-3]=NODAL,[4]=CELLCENTERED)`） |
| 每档壁面 DOF / 网格级别 | 第 2 节各表的面板数；DPanel 的 `gamma_dof` 与 `wall_panels`/`wake_panels` 见 2.4 |
| run exit 与日志 | `raw/lX/run.txt`（`exit=0` + 耗时 + 命令行）与 `stdout.log`；`ladders/*/stdout.log` |

## 4. 使用这些数字的硬约束

1. **panel 模式判断升力用 `CLwtot` / `trap.lod` 的条带载荷**。用户原始网格（`LECluster = 0.25`）上 `CLi` 会给出
   O(10¹)–O(10⁴) 的坏值（前缘聚类后出现尺寸突变的相邻排，逐条贴体涡条的 KJ 环量求和有 O(1/h) 奇异项）；
   `LECluster = 0.8` 的 3.55 网格上 `CLi` 正常（`CLi/CLwtot` ≈ 1）。
2. **`CDo` 是 VSPAERO 的加性经验项**（α=0 时 0.0205–0.0228，几乎与网格无关），所以 `CDtot`、`L/D`、`E` 不能当
   无粘阻力用；诱导阻力对照必须用 `CDiw`（见 7.2），或用 `trap.lod` 的 `Cd`。
3. **绝对值不能跨 mesher 版本引用**：同一 `.vspaero`、同一 exe，3.52 剖分的 6222 面板给 CL(5°) = 0.6544，
   3.55 剖分的 6344 面板给 0.7126（+8.9%），而几何差 <0.1%。收敛阶梯必须由同一个 mesher 生成。
4. **`NCPU` 字段无效**，线程数只能通过命令行 `-omp N`（1/8/32 线程 = 66.1 / 14.2 / 11.0 s，结果位级相同）。
5. **`trap.lod` 的 357 行是 3 个迎角 × 119 站**（文件无 AoA 列，按迎角顺序分块，每块 `ΣdSpan = 1.0`）；
   整表求和会得到 3 倍计数。
6. DPanel 展向环量用 PR #51 的物理展向积分口径：`elliptic_1028` 0.375065、`elliptic_1610` 0.375919；
   旧值 0.3944/0.3948 含约 5% 的 `ds/dy` 误差。

## 5. 复现

```powershell
# 0) 所有脚本都在 tools/vsp_evidence/（已入库）；数据目录 runs/vsp_conv 不入库
Invoke-Expression (Get-Content tools/vsp_evidence/evidence_rerun.ps1.txt -Raw)
# 1) 重建数值表
.conda_env/python.exe tools/vsp_evidence/evidence_build.py
# 2) 重新落盘原始文件 + 重建 sha256.json + 本 README 的表
.conda_env/python.exe tools/vsp_evidence/evidence_stage.py
.conda_env/python.exe tools/vsp_evidence/evidence_readme.py
# 3) 校验哈希
.conda_env/python.exe tools/vsp_evidence/verify_sha.py
```

3.55 的 panel/VLM 阶梯走 `vspapi`（见 `docs/VSPAPI.md`）：

```powershell
C:/Users/zyx20/anaconda3/envs/vsppytools/python.exe -m vspapi sweep <spec.json> --mode panel --ncpu 8
```

## 6. 边界

- VSPAERO 是独立势流对照，**不是实验真值**；本目录不声明物理精度验收，`physical_accuracy_accepted = false` 不变。
- 入库的是 3.55 阶梯（888–12996 面板 panel、420–6422 面 VLM）与 3.52 的 VLM 阶梯（240–12480 面）；
  用户原始 `cases/vsp/trap.vspgeom`（6249 面）的读数见 `docs/VSPAERO_PANEL_POLAR_DIAGNOSIS.md`。
- 粗档 `.vspgeom` 未入库（每档 0.1–6 MB），可用 `ladders/*/ladder.json` 里的 `tess_w`/`sect_tess_u` 一条命令重剖。

## 7. 复核后修正（2026-10-27，云端审阅）

### 7.1 归档完整性

135 条 `sha256.json` 里有 19 条曾被全局 `.gitignore` 的 `*.dat` / `*.log` 挡在 git 之外
（4×`raw/*/trap.tec.dat`、11×`ladders/*/stdout.log`、4×`raw/*/stdout.log`），干净 clone 只有 116 条。
已加 `docs/issue49_vspaero` 专属负规则并补提交（commit `490d43b`）；干净 clone 上 `verify_sha.py` 现在报 135/135。

### 7.2 `CLi` / `CLiw` / `CDiw` 的定义

OpenVSP 维护者说明：`CLi` 是 Biot–Savart 束缚涡环升力，`CLiw` 是尾迹涡量升力，`CLtot = CLo + CLi`、`CLwtot = CLo + CLiw`；
同理 `CDwtot = CDo + CDiw`。所以本目录的 `CLtot` 与 `CLwtot` 是**同一物理升力的两种分解**，不是厚翼 Cp 积分 vs 薄翼升力；
诱导阻力对照只能用 `CDiw`，`CDtot` 里的 `CDo` 是加性经验项。

### 7.3 最细档同一口径读数（5°）

| 方法 | CLtot | CLwtot | CDi | CDiw |
| --- | --- | --- | --- | --- |
| DPanel 12290 壁面 | 0.695979（CL_p） | 0.675226（CL_Γ） | — | 0.010276（CD_Trefftz） |
| VSPAERO panel 12996 面 | 0.704593 | 0.708639 | 0.018146 | 0.010723 |
| VSPAERO VLM 6422 面 | 0.639791 | — | — | 0.008718 |

### 7.4 截距与斜率比单点更可信

- 0°：DPanel 0.194801 vs VSP 厚 panel 0.222424（−12.4%）；5°：0.695979 vs 0.704593（−1.22%）。
- 0→5° 斜率每弧度：DPanel 5.7431，VSP 厚 panel 5.5252（+3.94%）。
- 结论：5° 的接近是**误差抵消**（截距偏低 + 斜率偏高），不能当作精度合格。

### 7.5 力矩必须换到同一参考点/参考弦

DPanel `cm = −0.048284`（`x_ref = 0.03888889`、`c_ref = 0.155556`）换算到 VSP 的 `(0, 0.2)`：

```
CM(0, 0.2) = [CMdp*0.155556 − 0.03888889*(CLdp*cosα + CDp*sinα)] / 0.2 = −0.172386
```

VSP `cm = −0.182497` → 统一后残差 **+0.010111**，不是未换算时的表观 +0.1342。

### 7.6 参考量不同

VSP 3.55 档：V = 100 m/s、rho = 0.002377、Re = 1e7；DPanel：V = 20 m/s、rho = 1.225（Re 未匹配）。
因此这是无量纲系数的对照，**不是同一雷诺/马赫**；厚 panel 是基本四边形面元，DPanel 是 Gmsh 三角形，面元数接近不代表分辨率相同。

### 7.7 截面 Cp 只覆盖 3.52 薄 VLM

`sections.json` 与 `raw/lX/trap.tec.dat` 都是 3.52 的 VLM 薄壳；**3.55 厚 panel 的上/下表面 Cp 截面尚未入库**。
要评价前缘吸力与压力恢复，需补 `ladders/api_panel_*` 的 tecplot 输出（y/b = 0.25 / 0.5 / 0.75），这是下一步。
