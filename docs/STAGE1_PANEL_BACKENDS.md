# Dpanel 阶段一：外部三维面元求解后端验收

状态：待本地安装与实测；不代表已集成。基线 master：`7b806439`，2026-10-24。

## 决策

第一阶段以**低马赫、定常、三维不可压面元法**为主，优先验证 VortexAD，FLOWPanel 作独立备选。保留现有 P2 全势实现为研究与参考，不继续将其作为整机生成器训练的前置条件。面元法可描述机翼与机身的联合无黏势流，但仍需 NeuralFoil/机身黏性闭合；不得声称它能预测真实失速或分离。

## 必须验证（按顺序）

1. **安装与官方算例**：VortexAD 官方 NACA0012 示例、FLOWPanel 官方 OpenVSP Cessna 多尾缘示例；记录完整命令、固定版本、环境、首次编译和热运行耗时、CPU/GPU、面元数和精度。
2. **前向输出**：原始 (C_p)、`panel_forces`、panel center/area/normal、(C_L,C_{Di},C_M) 和总力矩。禁止把截断压力当作训练真值。VortexAD 的非结构化后处理含 `Cp = maximum(Cp, cutoff, rho=100)` 路径；初步 smoke test 可记录默认设置，但不能视为原始压力验收。
3. **可微接口**：验证连续网格节点和迎角对任意局部载荷加权标量的 VJP。固定拓扑、三档中心差分对比局部面力和梯度。小规模 `JaxSimulator.compute_totals()` 可做诊断，但生成器接入应使用 VJP，避免物化所有面力的大 Jacobian。
4. **完整翼身**：共同暴露外表面，移除埋入机身的翼面和内部面，明确接缝与多尾缘；逐面压力/力/力矩与组件总量闭合。不得用单独机翼和机身系数相加冒充联合干扰。
5. **PyTorch 桥接**：前述门槛通过后再实现 `autograd.Function`；前向调用真实后端，反向调用已验证的固定拓扑几何 VJP。初始 probe 不等于生成器可用。

## 源码核对的风险

- VortexAD `PanelMethod` 基于 CSDL-alpha/JAX，提供局部面力与压力输出；默认后处理包含 (C_p) 截断。不得让截断或平滑静默进入正式目标。
- `insert_mesh` 本身不证明几何节点导数可用；必须把坐标作为计算图连续变量，并用中心差分核对局部力目标 VJP。
- VortexAD 的 `pyproject.toml` 未将 `csdl_alpha` 作为普通安装依赖，需固定兼容来源和版本。
- FLOWPanel 为 Julia 不可压面元法，官方整机示例包含 OpenVSP Cessna、多机翼/尾翼和机身；但必须核实所选 `RigidWakeBody` 类型、压力力、尾迹和 ImplicitAD 的设计变量导数。
- Julia 中支持自动微分不等于已存在 Python/PyTorch 的局部力 VJP。

## 本地 agent 安装命令

本地实际回执（PR #2 普通讨论评论，2026-10-24）：VortexAD smoke test 在 Python 3.11.17、VortexAD `ab6493981d5ff19721ba9fc42b6d015310d3d5cd`、CSDL-alpha `d4bf50ec91e722673b31849c4dfdad4c3076c715`、JAX 0.10.2 CPU 环境中 **1 passed，192.89 s**。首次直接 pip 安装失败的两个原因是：VortexAD wheel 未包含 STL/Msh 样例网格；其依赖声明没有安装 JAX（且 CSDL-alpha 依赖声明被注释）。通过源码 checkout 优先于 site-packages 的路径解决了样例数据问题。下方安装步骤针对该已复现结果修正。

在 Dpanel 仓库根目录执行；所有 checkout、环境和输出都留在 gitignored 的 `runs/backend_probes/`。不要覆盖现有 P2 环境。

```bash
mkdir -p runs/backend_probes/src
git clone https://github.com/LSDOlab/CSDL_alpha.git runs/backend_probes/src/CSDL_alpha
git -C runs/backend_probes/src/CSDL_alpha checkout d4bf50ec91e722673b31849c4dfdad4c3076c715
git clone https://github.com/LSDOlab/VortexAD.git runs/backend_probes/src/VortexAD
git -C runs/backend_probes/src/VortexAD checkout ab6493981d5ff19721ba9fc42b6d015310d3d5cd

python -m venv .venv-vortexad
source .venv-vortexad/bin/activate
python -m pip install -U pip
python -m pip install "jax[cpu]==0.10.2" pytest
python -m pip install -e runs/backend_probes/src/CSDL_alpha
python -m pip install -e runs/backend_probes/src/VortexAD
python -m pytest -q test/test_vortexad_backend.py -s
python -m pytest -q test/test_vortexad_mesh_vjp.py -s
```

Windows PowerShell 中激活命令改为 `.venv-vortexad\\Scripts\\Activate.ps1`。使用 editable source checkout 是为了确保 `SAMPLE_GEOMETRY_PATH` 能访问上游源码内的样例网格；单纯安装 wheel 当前不满足这个前置条件。若安装/checkout 失败，保存完整日志、命令与退出码，不删除环境，不宣称通过。

FLOWPanel 使用独立 Julia 项目/Depot：

```bash
mkdir -p runs/backend_probes/flowpanel_env runs/backend_probes/julia_depot
JULIA_DEPOT_PATH="$PWD/runs/backend_probes/julia_depot" julia --project=runs/backend_probes/flowpanel_env -e 'using Pkg; Pkg.add(["FLOWPanel", "GeoIO", "Rotations"]); using FLOWPanel, GeoIO, Rotations; println("FLOWPanel installed")'
JULIA_DEPOT_PATH="$PWD/runs/backend_probes/julia_depot" julia --project=runs/backend_probes/flowpanel_env test/probe_flowpanel.jl
```

## 验收与交付

- 安装和官方案例可复现，版本/commit 固定。
- 前向 (C_p)、面力、总力矩均有限，组件与整机力矩闭合。
- 迎角及至少一个连续网格形变变量的导数与三档中心差分一致。
- (C_p) 不依赖截断或平滑；若无法取得原始值，明确判为未通过。
- 报告冷/热耗时、峰值内存、CPU/GPU、面元规模、导数误差与失败回退。
- 输出放入 gitignored 的 `runs/backend_probes/`，不覆盖旧结果、不改 `aero.py` 默认路径、不启动生成器训练。
- 完成后只选一个训练主后端；如果两个都缺局部载荷几何 VJP，则记录阻塞原因并评估 VSPAERO 伴随或 Dpanel 自有离散路线。

## 本 PR 范围

本次只新增验收定义与 smoke test，不实现生产适配器。请本地 agent 回报 SHA、完整命令、退出码、traceback、安装版本、冷/热耗时、内存和建议选择。
