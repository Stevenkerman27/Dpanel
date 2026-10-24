# 高展弦比矩形 NACA0012 一致性算例

本批只运行 AR=8、20 的指定工况；M=0、alpha=+5°，各 AR 共用一张 P1/P2 曲面网格，先 node 后 weak 顺序求解，NCPU=8。每个求解器子进程上限 240 秒；超时保留日志并记录终止状态。此为高 AR 趋势与 2D 极限一致性检查，不是绝对 3D 精度认证。

几何为 build_mesh.py 生成的有限厚度、封闭后缘 NACA 多项式矩形翼。弦向 x、展向 y、厚度 z。AR8: span=8, chord=1, h=.2, curvature=32, wake strips=48, R=8, far-size=5/3。AR20: span=20, chord=1, h=.2, curvature=32, wake strips=120, R=15, far-size=2。网格生成过程中曲率细化可能使实际元素尺度小于 h。

## LLT 同展弦比预测

以下来自 `LL_RESULTS.md` 的 N=64 Prandtl 升力线结果。两种二维 section slope 是不同闭合假设，不将任何一个当作精确真值。

| AR | slope 模型 | CL | CDi |
|---:|---|---:|---:|
| 8 | 薄翼 2pi/rad | 0.422169241 | 0.007570878 |
| 8 | 二维 NACA0012 P2 slope | 0.452636769 | 0.008650571 |
| 20 | 薄翼 2pi/rad | 0.483828951 | 0.004328219 |
| 20 | 二维 NACA0012 P2 slope | 0.523634992 | 0.005020622 |

下表给出三维压力积分/环量升力相对同 AR 两种 LLT 预测的有符号差值（3D−LLT）。它们用于比较模型间距，不作为误差判据或调参目标。

| AR | Kutta | ΔCL pressure vs 薄翼 | ΔCL pressure vs NACA slope | ΔCL Gamma vs 薄翼 | ΔCL Gamma vs NACA slope |
|---:|---|---:|---:|---:|---:|
| 8 | node | +0.006260 | −0.024208 | +0.008600 | −0.021868 |
| 8 | weak | +0.008330 | −0.022138 | +0.010694 | −0.019774 |
| 20 | node | +0.027448 | −0.012358 | +0.028667 | −0.011139 |
| 20 | weak | +0.028201 | −0.011605 | +0.029426 | −0.010380 |

## 运行记录

### AR=8

- 网格 SHA256: `a1fc3d706b42147cf5450acaca05e23111084b9486cd61b09e32fa146aeabb38`
- 网格检查: 7/7 passed
- 节点/四面体: `55393` / `31754`

  - `dense_P2_Jacobian`: PASS
  - `all_nodes_used`: PASS
  - `cut_coordinate_pairing`: PASS
  - `wake_face_coordinate_pairing`: PASS
  - `TE_station_pairing`: PASS
  - `reference_element_reflection_pairing`: PASS
  - `all_exterior_faces_adjacency_and_midnodes`: PASS

- **node**: status=`converged_subsonic_unvalidated_lifting`, converged=`True`, wall time=15.28s, solver time=14.93029640000168, returncode=0, timeout terminated=False.
  - mesh nodes/tets=55393/31754; CL pressure/Gamma=0.42842892151592404/0.4307687742604971; CD pressure=0.009173420736074602; wall-normal RMS/max=0.03396718385727599/1.1617330241054311; Cp min/max=-1.8951864052613057/0.9998953033698409.
  - Raw wake diagnostics: `{"raw_paired_normal_mass_flux_jump_rms": 0.006536173982775411, "raw_paired_normal_mass_flux_jump_max": 0.15163889709356226, "raw_pressure_jump_rms": 0.0013220693652381164, "wake_upper_normal_velocity_rms": 0.05120886306817666, "model": "Fixed planar wake with recorded tilt; no free-wake tangency or pressure-continuity iteration", "conservation": "Paired raw gradients, separate from weak nodal continuity"}`
  - Final nonlinear residual=1.9305306488475436e-13; result=`results\aspect_ratio_v1\ar8\node\result.json`.

- **weak**: status=`converged_subsonic_unvalidated_lifting`, converged=`True`, wall time=15.10s, solver time=14.68698260000383, returncode=0, timeout terminated=False.
  - mesh nodes/tets=55393/31754; CL pressure/Gamma=0.43049880954893033/0.4328628813798389; CD pressure=0.009291895641203713; wall-normal RMS/max=0.034098319994729814/1.1623463417446374; Cp min/max=-1.89703081387364/0.9998846174870533.
  - Raw wake diagnostics: `{"raw_paired_normal_mass_flux_jump_rms": 0.006536173982775233, "raw_paired_normal_mass_flux_jump_max": 0.15163889709355916, "raw_pressure_jump_rms": 0.0015029778758703777, "wake_upper_normal_velocity_rms": 0.0520389252476639, "model": "Fixed planar wake with recorded tilt; no free-wake tangency or pressure-continuity iteration", "conservation": "Paired raw gradients, separate from weak nodal continuity"}`
  - Final nonlinear residual=1.9411762382355838e-13; result=`results\aspect_ratio_v1\ar8\weak\result.json`.

中剖面原始 Cp 对照文件：`results\aspect_ratio_v1\ar8\midspan_cp_comparison.csv`。3D 使用物理表面求积点 `|y|<0.1`，按 `z>0/<0` 分组；2D 使用保存的原始 Cp 上下表面点，在共同 x 范围线性插值到 3D 样本。它是采样/插值诊断，不是独立验证。

共同 x 样本上的 `Cp_3D−Cp_2D` RMSE：node 上/下表面为 `0.16793/0.06427`，weak 上/下表面为 `0.16746/0.06408`；绝对最大差分别约为 `0.79335/0.45943` 与 `0.79227/0.45850`。此差异较大，尤其上表面，故不支持已达到二维压力分布一致性。

### AR=20

- 网格 SHA256: `310b99109da77618a32774143b14daeead154af24f87971da3e10264dfa6411d`
- 网格检查: 7/7 passed
- 节点/四面体: `141634` / `80972`

  - `dense_P2_Jacobian`: PASS
  - `all_nodes_used`: PASS
  - `cut_coordinate_pairing`: PASS
  - `wake_face_coordinate_pairing`: PASS
  - `TE_station_pairing`: PASS
  - `reference_element_reflection_pairing`: PASS
  - `all_exterior_faces_adjacency_and_midnodes`: PASS

- **node**: status=`converged_subsonic_unvalidated_lifting`, converged=`True`, wall time=44.64s, solver time=44.1611133999977, returncode=0, timeout terminated=False.
  - mesh nodes/tets=141634/80972; CL pressure/Gamma=0.5112768223325738/0.5124955591840077; CD pressure=0.0067725806607326425; wall-normal RMS/max=0.02903502180802124/1.7385365182240717; Cp min/max=-2.426629430739894/0.9999640788592712.
  - Raw wake diagnostics: `{"raw_paired_normal_mass_flux_jump_rms": 0.004665931233054655, "raw_paired_normal_mass_flux_jump_max": 0.16881786654119857, "raw_pressure_jump_rms": 0.0006269594335024292, "wake_upper_normal_velocity_rms": 0.06682589133979956, "model": "Fixed planar wake with recorded tilt; no free-wake tangency or pressure-continuity iteration", "conservation": "Paired raw gradients, separate from weak nodal continuity"}`
  - Final nonlinear residual=1.1055500427029082e-12; result=`results\aspect_ratio_v1\ar20\node\result.json`.

- **weak**: status=`converged_subsonic_unvalidated_lifting`, converged=`True`, wall time=48.16s, solver time=47.636208500000066, returncode=0, timeout terminated=False.
  - mesh nodes/tets=141634/80972; CL pressure/Gamma=0.5120303520221665/0.5132549067007817; CD pressure=0.0068201946742210695; wall-normal RMS/max=0.029121801953000043/1.7386190832537975; Cp min/max=-2.427039471674324/0.9999637856502013.
  - Raw wake diagnostics: `{"raw_paired_normal_mass_flux_jump_rms": 0.004665931233054667, "raw_paired_normal_mass_flux_jump_max": 0.16881786654121456, "raw_pressure_jump_rms": 0.0007331724610440507, "wake_upper_normal_velocity_rms": 0.06735225142908989, "model": "Fixed planar wake with recorded tilt; no free-wake tangency or pressure-continuity iteration", "conservation": "Paired raw gradients, separate from weak nodal continuity"}`
  - Final nonlinear residual=1.1403930666971933e-12; result=`results\aspect_ratio_v1\ar20\weak\result.json`.

中剖面原始 Cp 对照文件：`results\aspect_ratio_v1\ar20\midspan_cp_comparison.csv`。3D 使用物理表面求积点 `|y|<0.1`，按 `z>0/<0` 分组；2D 使用保存的原始 Cp 上下表面点，在共同 x 范围线性插值到 3D 样本。它是采样/插值诊断，不是独立验证。

共同 x 样本上的 `Cp_3D−Cp_2D` RMSE：node 上/下表面为 `0.10137/0.03426`，weak 上/下表面为 `0.10145/0.03431`；绝对最大差分别约为 `0.56146/0.31606` 与 `0.56170/0.31624`。相较 AR8 有改善，但上表面差异仍显著，不能称为已通过二维极限检查。

## 资格说明

LLT 的 CL 是整翼积分预测，CDi 是升力线诱导阻力；求解器的 `cl_pressure`、`cl_jump` 和 `cd_pressure` 定义不同。需分别看压力积分、环量积分、raw wall normal 与 wake diagnostics，不能只用接近某个 LLT 数值来宣称验证。有限厚度几何、固定尾迹 cut、离散网格及不同 2D slope 均造成模型差异。本组结果只提供指定网格上的 AR 趋势和中剖面 2D 极限线索。

Runner SHA256: `63d1de3047099b23d7248c0632fe639d536cdd1c5f1bdc78d548e7a2ec8fa9f9`；总 wall time=143.48s。
