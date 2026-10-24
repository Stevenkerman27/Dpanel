# STEP闭体输入工作流

2026-10-23。`test_cad_workflow.py` 的13/13项预注册检查通过，原始数据在 `results/cad_workflow_v1/`。

| 输入 | 原生球半径/中心 | Cp面积RMS误差 | 原始壁面法向RMS | CL/CD最大绝对值 |
|---|---|---:|---:|---:|
| sphere | r=1，(2,-3,1) | 0.0090223 | 0.018622 | 6.34e-5 |
| sphere_large | r=1000，(2000,-3000,1000) | 0.0091644 | 0.018931 | 8.35e-6 |

球面压力参照为无限域不可压解析解，实际外盒用均匀Dirichlet，有限域误差仍包含其中。两例分别重网格，因此这不是逐节点单位等变检验；它是实际CAD读取、归一化、网格、求解和原始压力检查的尺度工作流试验。

另一个开口平面STEP没有实体体积，返回退出码2及 `CAD input contains no closed solid volumes`，不尝试缝合或将开面伪装成封闭物面。

归一化由CAD包围盒中心和最大边长定义，保留在 `mesh.json` 的 `native_origin/native_length`。网格和求解输出坐标为归一化坐标；原生坐标=`native_origin+native_length*points`。面积和Sref须按长度平方转换。测试的原生映射相对误差约1e-7/1e-10，包含OpenCASCADE包围盒容差。

统一入口示例：`python aero.py --case cases/cad_sphere.json --out runs/new_cad_case`。例子读取本次工作流生成的STEP球体；自己的CAD修改 `geometry.cad`，路径相对于case JSON。`numerics.sref`用归一化面积单位。详细边界见 `CAD_PROTOCOL.md`。

本入口支持OpenCASCADE闭合实体导入，但没有认证任意复杂CAD、尖缝/小间隙、混合部件或含封闭流体腔。它不包含环量、后缘或尾迹，不能通过将机翼STEP放入此入口来代替三维升力解。压力与数值诊断完整保留，仍需逐几何验证。
