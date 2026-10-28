# tools/vsp_evidence

issue #49 的「梯形翼 DPanel vs VSPAERO 同几何对比」实验脚本。**产物已入库**在
`docs/issue49_vspaero/`（原始 polar / lod / history / Cp / 日志 + 逐文件 SHA-256）；
工作数据在 `runs/vsp_conv/`，该目录被 `.gitignore` 忽略，需按下表重跑生成。

## 脚本

| 脚本 | 作用 |
| --- | --- |
| `evidence_build.py` | 读 `runs/vsp_conv/{vlm_lX,evidence/lX,dp_s0XX}`，写 `docs/issue49_vspaero/{levels,sections,dpanel}.json` |
| `evidence_stage.py` | 把原始文件拷进 `docs/issue49_vspaero/{raw,ladders,dpanel,figs}/`，写 `sha256.json` |
| `evidence_readme.py` | 从 staged JSON 生成 `docs/issue49_vspaero/README.md`（表由数据生成，不会漂移） |
| `verify_sha.py` | 校验 `sha256.json`；全部命中才 exit 0 |
| `inspect_case.py` | 打印一个 VSPAERO 案例目录的 mesh/polar/history/lod 摘要与健康告警 |
| `dpanel_case.py` | 驱动 `panel.cad_half` + `solve_half_fields`，对每个迎角出 CL_p/CL_Γ/CD_p/CD_Trefftz/CM_y |
| `make_trap_cad.py` | 造梯形翼 STEP + `trailing_edge.json`（NACA 2412、root 0.2 / tip 0.1、半展 1.0） |
| `analyze.py` | 汇总 DPanel 五档 + VSPAERO 阶梯 → `summary.json` / `summary_table.md` / `span_loading.npz` |
| `figures.py` | fig16–fig20（收敛、展向载荷、迎角扫掠、尾迹、Cp 图） |
| `api_ladder.py` | 走 `vspapi` 生成 OpenVSP 3.55 的 panel/VLM 同 mesher 阶梯（`--mode panel --ncpu 8`） |
| `evidence_rerun.ps1.txt` | 3.52 exe 逐档单迎角重跑（`WriteTecplotFile=1`），`Invoke-Expression (Get-Content ... -Raw)` 执行 |

## 顺序

```powershell
Invoke-Expression (Get-Content tools/vsp_evidence/evidence_rerun.ps1.txt -Raw)  # 1) 重跑原始输出
.conda_env/python.exe tools/vsp_evidence/evidence_build.py                      # 2) 数值表
.conda_env/python.exe tools/vsp_evidence/evidence_stage.py                      # 3) 落盘 + SHA
.conda_env/python.exe tools/vsp_evidence/evidence_readme.py                     # 4) README 表
.conda_env/python.exe tools/vsp_evidence/verify_sha.py                          # 5) 校验
```

`evidence_build.py` 必须在 `evidence_stage.py` **之前**跑：两者都写
`docs/issue49_vspaero/`，stage 会以磁盘实际内容重建 `sha256.json`，因此它必须最后落盘。

## 数据前置条件（不入库）

- `cases/vsp/trap.vspgeom`：用户给的 6249 面板厚网格（入 `docs/issue49_vspaero` 的 SHA 清单）；
- `runs/vsp_conv/vlm_l{1..4}/`：DEP-OPT 路径的 VLM 阶梯（`trap.polar`），由 `dep_vsp.py` 生成；
- `runs/vsp_conv/evidence/l{1..4}/`：3.52 exe 的单迎角 5° 重跑（`-omp 8`）；
- `runs/vsp_conv/dp_s0{20,14,10,07,05}/`：DPanel 五档（`case.json` + `mesh/surface.json`）；
- `runs/vsp_conv/api_{panel,vlm}_*/`：OpenVSP 3.55 阶梯（`ladder.json`），由 `api_ladder.py` 生成。

DPanel 侧运行环境：`.conda_env`（Py3.11 / JAX 0.10.2 CPU），worktree `runs/pr49_git`（commit `535f907`），
`TMP/TEMP/TMPDIR=runs/pipwork`。VSP 侧运行环境：`C:\Users\zyx20\anaconda3\envs\vsppytools`（OpenVSP 3.55.0）。

## 相关文档

- `docs/VSP_TRAPEZOID_CONVERGENCE.md`：三方法收敛结论与归因；
- `docs/VSPAERO_PANEL_POLAR_DIAGNOSIS.md`：panel 模式 `CLi` 异常（`LECluster` 根因）；
- `docs/VSPAPI.md`：`vspapi` 包与模式规则；
- `docs/issue49_vspaero/README.md`：证据包索引（本仓库里的规范副本）。