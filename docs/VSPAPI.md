# VSP API 代码收敛：本项目 / DEP-OPT / GRASS / aircraft-tools

| | |
| --- | --- |
| 结论 | 新增 `vspapi/` —— 本项目内唯一一份 OpenVSP/VSPAERO 客户端（约 800 行 + 17 个纯 Python 测试），替代 `runs/vsp_conv/` 下的散装脚本与 DEP-OPT 整份拷贝 |
| 依赖 | 纯层（tessellate/config/mesh/results/health）零依赖；只有 `vspapi.api` 需要 OpenVSP。求解器环境 `.conda_env` 不装 OpenVSP 也能跑全部测试 |
| 状态 | `physical_accuracy_accepted = false`（外部对照工具，不改变求解器默认） |

## 1. 收敛前的实际状况

同一条代码血统在本机有 **4 份拷贝**：

| 位置 | 大小 | 来源 / 状态 |
| --- | --- | --- |
| `D:/3D/Projects/DEP-OPT/infrastructure.py` | 93 kB | 母本，生产验证过（机翼/机身/螺旋桨/控制面/FFD/质量/XFOIL 全在内） |
| `D:/3D/Projects/GRASS-plane/infrastructure.py` | 53 kB | GRASS 自己的分支（注释写 "for vsp3.41"），用 `vsp.SET_SHOWN` 当 thick/thin 集 |
| `D:/3D/Projects/aircraft-tools/aircraft_tools/openvsp_infrastructure.py` | 52 kB | 从 GRASS 提出来准备共享，但 DEP-OPT 与本项目都没用它 |
| `panel/runs/vsp_conv/depopt/` | 93 kB | 上一轮照抄的 DEP-OPT 副本，仅供 `dep_vsp.py` 用 |

本项目里另有三层散装代码：

* `audit/run_case.py`、`audit/junction/run_experiments.py`、`audit/reference_wing3d/run_reference.py` —— 早期 WBCAero 对照，各自 `import openvsp`。
* `runs/vsp_conv/*.py`（`dep_vsp.py`、`panel_wake.py`、`panel_ladder.py`、`cluster_probe.py`、`vsp_case.py` …）—— 一次性驱动脚本。
* `vspapi` 之前的一次性 API 内省脚本（`runs/_probe_*.py`、`runs/_enum*.py`）—— 结论已固化进 `vspapi.api.geometry_sets()` 与 `test/test_vspapi_core.py`；脚本本身在 2026-10-27 清理时删除，见 `docs/CLEANUP_2026-10-27.md`。

**关键事实：`panel/` 求解器本身零 VSP 代码。** 所有 VSP 使用都在外部对照与审计里，所以可以整体收进一个独立包而不碰求解器（没有循环依赖）。

## 2. 新包 `vspapi/`

```
vspapi/
  tessellate.py   翼面分块（next_tess_value / make_wing_tess_plan / wing_area）   <- DEP-OPT 逐行移植
  config.py       SweepConfig <-> .vspaero（字节稳定，可回读）
  mesh.py         .vspgeom 读写（兼容 3.55 多一行面元计数）、面积/质量/边界审计、y=0 焊接镜像
  results.py      .polar / .history / .lod（按迎角分块）/ .tec.dat
  health.py       LECluster 门 + CLi/CLwtot 一致性门
  api.py          Case：建翼、compute_geometry(mode)、run(cfg)、assert_mode   <- 唯一 import openvsp
  cli.py          python -m vspapi version | inspect | sweep
  README.md       血统、适配点、迁移说明
test/test_vspapi_core.py   17 passed（0.12 s，无需 OpenVSP）
```

移植范围（DEP-OPT -> vspapi）：`next_tess_value`、`make_wing_tess_plan`、
`create_wing`（含 `SetSetFlag` 进 thick 集）、`_set_vspaero_geometry_sets`、
`_run_vspaero_compute_geometry`、`_run_vspaero_sweep`、`_read_vspaero_polar`、
`_read_vspgeom_mesh_quality`。

**没有**移植：螺旋桨/作动盘、机身、控制面、FFD、质量与强度、XFOIL 耦合、
`case_name` 全局状态、工程参数表。这些留在 DEP-OPT。

## 3. 为 panel 做的两处适配

### 3.1 模式显式化（否则 panel 静默退化成 VLM）

VSPAERO 没有分析方法开关，模式由**几何集**决定：

| mode | `GeomSet` | `ThinGeomSet` | 产出网格 | 湿面积/平面面积 | stdout `Model type` |
| --- | --- | --- | --- | --- | --- |
| `panel` | `SET_FIRST_USER`(3) | `SET_NONE`(-1) | 双面厚体 | ~1.02 | `PANEL or MIXED` |
| `vlm` | `SET_NONE`(-1) | `SET_FIRST_USER`(3) | 单层薄壳 | ~0.50 | `VLM` |

两个坑，都是实测踩出来的：

* **不在任何集里的机翼一律按 VLM 剖分**，`GeomSet=SET_ALL`(0) 也不代表 panel。
  所以 `Case.add_wing` 和 DEP-OPT 的 `create_wing` 一样调用
  `SetSetFlag(wid, SET_FIRST_USER, True)`，再由
  `Case.assert_mode()` 用「湿面积 / 全机平面面积 ~ 1.02（panel）或 ~ 0.50（VLM）」
  事后复核——这条判据现在能挡住两类静默退化。
* 机翼同时进 thick 和 thin 两个集会被**剖分两次**，VSPAERO 打
  `In both the thick and the thin geometry set, so meshed twice`。

### 3.2 聚类门（`LECluster`）

panel 模式的 `CLi`（后缘 Kutta-Joukowski 环量求和）在前缘弦向聚类过紧时会错到
100 倍以上（证据见 `docs/VSPAERO_PANEL_POLAR_DIAGNOSIS.md` 第 4.3 节）。其余设置
全同时实测：

| `LECluster` | `CLi`@5deg | `CLwtot`@5deg |
| --- | --- | --- |
| 0.25 | 125.45 | 0.6492 |
| 0.50 | 5.5015 | 0.6596 |
| 0.80 | 0.6346 | 0.6544 |

`add_wing` 默认 `LECluster=0.8 / TECluster=0.9`，低于 0.5 直接拒绝
（`allow_risky_cluster=True` 可越过）；`check_polar_health` 解完后逐行检查
`|CLi/CLwtot| > 1.5`，并把可信列限定为 `CLwtot/CDwtot`。

## 4. 环境坑

### 4.1 vsppytools 里的 3.55 装坏了（目录 ACL）

装 3.55.0 之后 `site-packages/openvsp` 与 `openvsp-3.55.0.dist-info` 两个目录
**拒绝读取**（`PermissionError: [Errno 13]` / `WinError 5`），于是
`import openvsp` 会成功但拿到**空命名空间包**（`__file__ is None`、0 个属性、
没有 `GetVSPVersion`）。这是 pip 写入时的 ACL 问题，不是包本身的问题。

`vspapi.api.load_openvsp()` 因此两级兜底：先正常 `import`，若模块缺
`GetVSPVersion` 就判为坏包；再按 `VSP_ROOT` / 仓库目录 / 上级目录里的
`OpenVSP-*-win64`（版本号从新到旧）从 `<root>/python/` 导入，并
`os.add_dll_directory(<root>)`。实测 `OpenVSP 3.55.0` 可用。彻底修法是用管理员
权限重装该包，或修正那两个目录的 ACL。

### 4.2 .vspgeom v3 在 3.55 多了「面元计数行」

3.55 的 `VSPGeomFileName` 在点表与面元记录之间插了一行单独的面元数，并在文件
末尾追加每面 4 行的厚度块（3.52 没有）。`vspapi.mesh.read_vspgeom` 两种布局都
接受（`extra_lines` 记录尾巴行数）。旧文件回归：`pth_52x120/trap.vspgeom`
（6269 点 / 6222 面 / 面积 0.3066）与 `cases/vsp/trap.vspgeom`（6306 / 6249 /
0.3070）解析结果与 DEP-OPT 自己的读取器一致。

## 5. 验收：与 DEP-OPT 的几何路径逐字节等价

DEP-OPT 的 `create_wing` 建出的模型（清理时的一次性脚本调用它，等价做法：先用 `vspapi.api.load_openvsp()` 修好 3.55 的坏包，再 `import infrastructure as infra` → `infra.create_wing(...)` → `infra._run_vspaero_compute_geometry(...)`）交给 3.55 剖分，与 `vspapi` 的 `WingSpec(spanlist=[1.0], chordlist=[0.2, 0.1], tess_w=53, sect_tess_u=[121])` 剖分，**输出文件字节相同**：

```
SHA-256  D7E13757EE8355A92724BB130DA1D12C7E331B039A34301D1E37C222B49CBB9B
         runs/vsp_conv/pth_355_re/trap.vspgeom      (DEP-OPT create_wing, 3.55 mesher)
         runs/vsp_conv/api_355_panel/trap.vspgeom   (vspapi WingSpec,     3.55 mesher)
         6344 panels / 6367 points / wetted area 0.30662 / planform 0.30   -> ratio 1.022 (panel)
```

即「移植」没有引入几何差：`vspapi` 的 wing 就是 DEP-OPT 那只翼。实求解（5°，单点）`CLwtot 0.71260 / CLi 0.7105`，与 `runs/vsp_conv/api_ladder.py` 的 52×120 档（0.71256）一致到 5e-5。

**顺带定下的两条事实**：

1. **同一个模型、同一个 exe，只换 mesher 版本，CL 差 8.9%**：3.52 剖分 6222 面板 → `CL(5°) 0.6544 / CL(0°) 0.1695`；3.55 剖分 6344 面板 → `0.7126 / 0.2259`。两者体积 1.8124e-3 vs 1.8134e-3、湿面积 0.31807 vs 0.31829、体积质心 z 0.0025436 vs 0.0025422（差都 <0.1%），差异是一个纯截距（斜率同为 5.6 /rad）。结论：**panel 模式的收敛阶梯必须由同一个 mesher 生成，绝对 CL 不宜跨版本引用**。
2. **NCPU 有效，但通道是命令行 `-omp N`**（2026-10-27 更正）：`vspaero.exe` 自带 `Usage: vspaero [options] <FileName>` 与 ` -omp <N>   Use 'N' processes.`，内部调用 `omp_set_num_threads`。同一个 6344 面板厚面元算例（5°，`.vspaero` 完全不变，CL 都 0.71260）：

   | 调用 | 线程 | 墙钟 |
   | --- | --- | --- |
   | `vspaero trap` | 1 | 66.1 s |
   | `vspaero -omp 8 trap` | 8 | **14.2 s**（4.65×） |
   | `vspaero -omp 32 trap` | 32 | **11.0 s**（6.0×） |

   **两条错误通道**（都会静默退化成 1 线程，exe 里根本没有 `NCPU` 字符串）：`.vspaero` 里写 `NCPU = 8`（无效）、环境变量 `OMP_NUM_THREADS=8`（VSPAERO 自己 `omp_set_num_threads(1)`，也无效）。OpenVSP 的 API 之所以「NCPU 有用」，是它在启动 exe 时把 `NCPU` 转成了 `-omp`。`Case.run(ncpu=N)` 现在就是这样拼命令行的，并把 exe banner 回读成 `omp_threads` 字段（应等于 N）。`OMP/OPENBLAS/MKL_NUM_THREADS` 仍会被设上，但只用于限制 BLAS 线程，避免多案例并发时 `OpenBLAS error: Memory allocation still failed after 10 retries`。

## 6. 迁移到 aircraft-tools

`vspapi` 刻意做到无全局状态、纯层零依赖，就是为了整包搬到
`D:/3D/Projects/aircraft-tools`（改名 `aircraft_tools.vsp` 或平级模块）。
搬过去之后 DEP-OPT 与 GRASS 改成 `import` 它，本机就只剩一份实现。
在那之前，`runs/vsp_conv/depopt/` 保留作历史证据，但新代码一律走 `vspapi`。

## 7. 复现

```powershell
# 纯层测试（求解器环境即可，无需 OpenVSP）
$env:PYTHONPATH = 'D:/3D/Projects/panel'
& .conda_env/python.exe -m pytest test/test_vspapi_core.py -q -p no:cacheprovider

# 实际求解（需要 OpenVSP 环境）
$env:VSP_ROOT = 'D:/3D/Projects/OpenVSP-3.55.0-win64'
$env:PYTHONPATH = 'D:/3D/Projects/panel'
& C:/Users/zyx20/anaconda3/envs/vsppytools/python.exe -m vspapi version
# 求解（线程数只能走 -omp，Case.run 已自动拼上）
& C:/Users/zyx20/anaconda3/envs/vsppytools/python.exe -m vspapi sweep runs/vsp_conv/api_355_spec.json
# 手工复现线程效果：& $VSP_ROOT/vspaero.exe -omp 16 trap   （cwd 需含 trap.vspgeom + trap.vspaero + trap.vkey）
```