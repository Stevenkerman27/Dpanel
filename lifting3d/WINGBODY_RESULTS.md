# 厚翼身组合体鲁棒性矩阵

## 预注册协议

目标是检查现有 P2 全势求解流程在厚度、机身半径及后缘邻近几何变化下的数值鲁棒性。结果属于指定离散模型的实现诊断，不是绝对气动精度认证；求解器返回收敛不代表压力场通过。

### 固定模型与网格参数

- 机翼：矩形、对称 NACA 多项式截面，弦长 1，厚度按矩阵变化；弦向 x、展向 y、厚度 z。
- 机身：OCC 椭球，与翼体执行实体布尔合并；长度 3，`body_aft` 按表变化。
- 外流体域：远场半径 `R=5`，`far-size=5/3`。
- 网格：`size=0.2`、曲率细化 32、wake strips 24；不加隐式倒圆，不改几何。
- 流动：弱 Kutta；主矩阵均为 `M=0, alpha=+5°`。每个 solver 子进程最多 240 秒，NCPU=8；全流程串行。超时必须终止并确认子进程退出，保存部分日志及 timeout 状态。
- 对照：对 `(t=.24, body_radius=.35, body_aft=.9)` 复用同一张网格，追加 `M=0, alpha=0/-5°` 与 `M=.2, alpha=+5°`。

| Case | 厚度 | 机身半径 | 机身后端 x |
|---|---:|---:|---:|
| `t012_r020_a090` | .12 | .20 | .90 |
| `t024_r035_a090` | .24 | .35 | .90 |
| `t036_r035_a090` | .36 | .35 | .90 |
| `t024_r015_a090` | .24 | .15 | .90 |
| `t024_r050_a090` | .24 | .50 | .90 |
| `t024_r035_a099` | .24 | .35 | .99 |

### 预定检查和门限

1. 每张网格运行现有 `test_mesh.py` 的 7 项检查；任何失败原样保留并继续下一几何，不重网格调参。
2. 每个成功解运行现有 `analyze.py`。保存 junction 与完整 wall 分区的面积、raw normal RMS/max、Cp min/max，全局 raw normal 峰值位置、Cp 最小值位置、按 tag 原始质量通量、wake 通量/压力跳跃、压力/环量升力差和 component sum error。
3. 所有 `field.npz` 原始 Cp 必须逐项有限；只报告有限性，不裁剪、不过滤极值、不以有限值等同物理正确。
4. 每例组件力矢量之和与全壁面压力力差的最大分量误差门限为 `1e-10`。该门限是实现闭合检查。
5. 基准网格 `alpha=0` 的 `|CL_pressure|`、`|CL_gamma|`、`max|Gamma|` 门限均为 `1e-9`。
6. 基准网格 `alpha=+5°/-5°` 的压力 CL 和环量 CL 之和绝对值，以及展向 Gamma 逐点反对称最大误差，门限均为 `1e-9`。
7. `M=.2` 个案记录亚声速状态、最小 beta、最大局部 Mach 和完整 raw wall/wake 诊断，不预设其载荷必须接近 M=0。
8. CAD/布尔失败、网格检查失败、solver 失败或超时均作为失败证据保留；不改上述门限，不因失败补做参数扫掠。

Cp 的绝对分布没有预设接受窗口，因此即使代数残差、有限性及载荷闭合通过，也不能称作 Cp 通过或气动精度认证。

## 执行结果

执行结果已附于本协议之后。完整输入、日志、mesh、field、result 与分析输出保存在 `results/wingbody_matrix_v1/`。

## 执行结果

### 通过率与门禁汇总

- 几何/CAD 构建并通过 `test_mesh.py`：**6/6**；每张网格 7 项全通过。不存在 CAD、布尔或网格失败。
- Solver 方程收敛：**9/9**，超时 **0/9**；所有 mesh 使用弱 Kutta。Solver 收敛仅是代数收敛状态。
- 预注册的 Cp 有限性和组件压力力闭合：**9/9** 均通过；所有 9 个 raw Cp 值集合有限，组件力矢量闭合最大误差 `1.32e-14 < 1e-10`。
- 固定基准网格 `alpha=0` 零载荷门限通过；压力 CL/Gamma CL/max|Gamma| 误差分别为 `5.94e-15/2.94e-16/4.58e-15`。
- 固定基准网格 `alpha=±5°` 反对称门限通过；压力 CL/Gamma CL/Gamma 逐点最大误差分别为 `6.83e-15/1.22e-15/1.37e-14`，均小于 `1e-9`。
- `M=.2, alpha=5°` 解收敛；`min_beta=0.96564`、最大局部 Mach=`0.46837`，保留为该模型工况的亚声速诊断。
- 所有 9 个解所用共享 `p2.py` SHA256 一致，为 `920f88a737414a3d3a135d91a8373f877dec4619986e7452654f6e33619aa44a`，与本批启动前冻结版本相同。

以下 raw wall/wake 数值没有预设通过门限，仅报告观测：wall-normal RMS 范围 `0.0243–0.0408`，全壁面最大绝对法向速度 `0.789–1.195`；raw wake 配对法向通量跳跃 RMS `0.00934–0.01969`、最大值 `0.153–0.295`；raw wake Cp 跳跃 RMS `8.99e-15–0.01155`。这表明几何/代数门禁通过不等于 wall impermeability 或 wake 条件已通过。raw Cp 最小值范围 `−4.653…−2.575`、最大值 `0.9999…1.0100`，不裁剪、不据此宣称压力精度通过。

| Case | 厚度 | 机身半径 | aft | Mesh 状态 | points/tets | mesh checks |
|---|---:|---:|---:|---|---:|---:|
| t012_r020_a090 | 0.12 | 0.20 | 0.90 | solved | 118198/74706 | 7/7 |
| t024_r035_a090 | 0.24 | 0.35 | 0.90 | completed_with_gate_results | 113012/72426 | 7/7 |
| t036_r035_a090 | 0.36 | 0.35 | 0.90 | solved | 106684/69358 | 7/7 |
| t024_r015_a090 | 0.24 | 0.15 | 0.90 | solved | 125639/78418 | 7/7 |
| t024_r050_a090 | 0.24 | 0.50 | 0.90 | solved | 106232/69050 | 7/7 |
| t024_r035_a099 | 0.24 | 0.35 | 0.99 | solved | 114059/73104 | 7/7 |

## 各例结果

### t012_r020_a090

- mesh SHA256 `7d1ebbb4c8bc7fb9f1aa5786dab8e90d833ccb7dd821541b62ca1a5d24d36e70`; dense minimum Jacobian `7.503751265753941e-07`; mesh metadata `results/wingbody_matrix_v1/t012_r020_a090/mesh/mesh.json`.

- **m0_a5** (`M=0.0`, `alpha=5.0°`, weak): status=`converged_subsonic_unvalidated_lifting`, converged=`True`, wall=46.17s, timed_out=False, timeout child terminated=False.
  - nodes/tets=118198/74706; CL pressure/Gamma=0.34624821958479785/0.3504185298372068; CD pressure=0.0107498463862706; raw Cp finite=311652/311652 (min/max=-2.574504233072527/0.9998549156926049).
  - wall RMS/max=0.03204729436286756/0.9226696779269938; global raw normal peak={'value': -0.9226696779269938, 'location': [0.00014328252064083088, 1.5993241064349835, -0.0016397714899849707], 'region': 'leading'}; global Cp minimum={'value': -2.574504233072527, 'location': [0.005542716394270901, -0.18392864331259182, 0.012920576869038062], 'region': 'junction'}.
  - junction/full-wall regions=`{"interior": {"area": 5.962472358375241, "normal_rms": 0.009775974665411939, "normal_max": 0.08847942504662726, "cp_min": -1.0115960115225247, "cp_max": 0.2568836886698933}, "junction": {"area": 0.7983039769350087, "normal_rms": 0.018136362002344843, "normal_max": 0.5734853357063963, "cp_min": -2.574504233072527, "cp_max": 0.9991461360017512}, "leading": {"area": 2.993806422578679, "normal_rms": 0.02810973518925392, "normal_max": 0.9226696779269938, "cp_min": -1.9430125778701557, "cp_max": 0.9998549156926049}, "tip": {"area": 0.5718407695415632, "normal_rms": 0.11086951404609369, "normal_max": 0.8438869839466281, "cp_min": -1.5153273405644363, "cp_max": 0.9880451908860017}, "trailing": {"area": 0.35313716630514225, "normal_rms": 0.04581163206201865, "normal_max": 0.10191113505906008, "cp_min": -0.050600911834073115, "cp_max": 0.35433941113931555}}`.
  - component loads=`{"20": {"force_coefficient": [0.002056203817552903, -2.7631291591416267e-06, 0.033318313500924644], "cl": 0.03301231728802036, "cd": 0.004952261701514624, "area": 2.81677035320645, "normal_rms": 0.004610863408695861}, "30": {"force_coefficient": [-0.021524784595217165, -0.0001481124709343438, 0.3125492379193692], "cl": 0.3132359022967707, "cd": 0.005797584684754931, "area": 7.862790340529184, "normal_rms": 0.03724696394514303}}`; component sum error=6.938893903907228e-15 (pass=True).
  - raw mass by tag=`{"10": 0.0, "20": -0.0004411664247609558, "30": -0.04169388408869265, "100": -8.122821820322201e-05, "201": -0.5897929521074283, "202": 0.6099211193540451}`; wake diagnostics=`0.009342048104679718` and `{"raw_paired_normal_mass_flux_jump_rms": 0.009342048104679718, "raw_paired_normal_mass_flux_jump_max": 0.15325281211126374, "raw_pressure_jump_rms": 0.0028444913225822857, "wake_upper_normal_velocity_rms": 0.05864666204772813, "model": "Fixed planar wake with recorded tilt; no free-wake tangency or pressure-continuity iteration", "conservation": "Paired raw gradients, separate from weak nodal continuity"}`.
  - result=`results\wingbody_matrix_v1\t012_r020_a090\m0_a5\result.json`; analysis=`results\wingbody_matrix_v1\t012_r020_a090\m0_a5\analysis.json`.

### t024_r035_a090

- mesh SHA256 `dd0a654124aaa7e428d28906ad1e9173f78a013ada50fab69a35f3077841ab0b`; dense minimum Jacobian `6.509631251128634e-07`; mesh metadata `results/wingbody_matrix_v1/t024_r035_a090/mesh/mesh.json`.

- **m0_a5** (`M=0.0`, `alpha=5.0°`, weak): status=`converged_subsonic_unvalidated_lifting`, converged=`True`, wall=51.87s, timed_out=False, timeout child terminated=False.
  - nodes/tets=113012/72426; CL pressure/Gamma=0.3792998478824885/0.3891329409453903; CD pressure=0.012698049148691139; raw Cp finite=276804/276804 (min/max=-4.07767565463395/0.999956242653711).
  - wall RMS/max=0.03028126802044346/0.8158094725956095; global raw normal peak={'value': 0.8158094725956095, 'location': [0.002621593332195602, 1.999822049379482, -0.017828827331825385], 'region': 'tip'}; global Cp minimum={'value': -4.07767565463395, 'location': [0.0017151777891971096, 1.9998542224216567, -0.014492342342605696], 'region': 'tip'}.
  - junction/full-wall regions=`{"interior": {"area": 6.078498633125613, "normal_rms": 0.014776322612548754, "normal_max": 0.13000550248647427, "cp_min": -1.5368214834813516, "cp_max": 0.43340898705666386}, "junction": {"area": 0.883447139711817, "normal_rms": 0.014858282583319683, "normal_max": 0.19056863716483285, "cp_min": -1.903192548330093, "cp_max": 0.9988838370760066}, "leading": {"area": 4.885557315353157, "normal_rms": 0.010034415080401243, "normal_max": 0.18758197721075043, "cp_min": -1.6575245491616806, "cp_max": 0.999956242653711}, "tip": {"area": 0.7507977613442112, "normal_rms": 0.09940937515225444, "normal_max": 0.8158094725956095, "cp_min": -4.07767565463395, "cp_max": 0.9682326378850588}, "trailing": {"area": 0.33427319130688443, "normal_rms": 0.08517186451720173, "normal_max": 0.23765358696134925, "cp_min": -0.02061546174306761, "cp_max": 0.6244355524210085}}`.
  - component loads=`{"20": {"force_coefficient": [0.009909234669813013, 8.9002367130432e-06, 0.06381429571718301], "cl": 0.0627078163482081, "cd": 0.01543330938136442, "area": 4.949711709019088, "normal_rms": 0.004219957694474861}, "30": {"force_coefficient": [-0.03031766539805095, -1.1987790725108353e-05, 0.31514890963536407], "cl": 0.3165920315342864, "cd": -0.0027352602326731652, "area": 7.982862331822595, "normal_rms": 0.03839875507006966}}`; component sum error=5.995204332975845e-15 (pass=True).
  - raw mass by tag=`{"10": 0.0, "20": -0.0008412200012809664, "30": -0.06966376038390588, "100": -0.0037289464798497818, "201": -0.49503973202119017, "202": 0.529160433887609}`; wake diagnostics=`0.015287102725524972` and `{"raw_paired_normal_mass_flux_jump_rms": 0.015287102725524972, "raw_paired_normal_mass_flux_jump_max": 0.25559803371760875, "raw_pressure_jump_rms": 0.007507983953646233, "wake_upper_normal_velocity_rms": 0.0607696519347558, "model": "Fixed planar wake with recorded tilt; no free-wake tangency or pressure-continuity iteration", "conservation": "Paired raw gradients, separate from weak nodal continuity"}`.
  - result=`results\wingbody_matrix_v1\t024_r035_a090\m0_a5\result.json`; analysis=`results\wingbody_matrix_v1\t024_r035_a090\m0_a5\analysis.json`.

- **m0_a0** (`M=0.0`, `alpha=0.0°`, weak): status=`converged_subsonic_unvalidated_lifting`, converged=`True`, wall=50.45s, timed_out=False, timeout child terminated=False.
  - nodes/tets=113012/72426; CL pressure/Gamma=-5.936649911479111e-15/-2.9381880367747704e-16; CD pressure=0.002267919919105703; raw Cp finite=276804/276804 (min/max=-3.840503057489429/0.9999483166818915).
  - wall RMS/max=0.02425328998850382/0.7894793350620118; global raw normal peak={'value': 0.7894793350620118, 'location': [0.002621593332195602, 1.999822049379482, -0.017828827331825385], 'region': 'tip'}; global Cp minimum={'value': -3.840503057489429, 'location': [0.0017151777891971096, 1.9998542224216567, -0.014492342342605696], 'region': 'tip'}.
  - junction/full-wall regions=`{"interior": {"area": 6.078498633125613, "normal_rms": 0.01395123215207658, "normal_max": 0.06312051630641036, "cp_min": -0.9298966346879765, "cp_max": 0.4209093338383275}, "junction": {"area": 0.883447139711817, "normal_rms": 0.014504874416350672, "normal_max": 0.1755937256623249, "cp_min": -0.8803731555764485, "cp_max": 0.9959706781723624}, "leading": {"area": 4.885557315353157, "normal_rms": 0.009469748128684088, "normal_max": 0.15834686491496447, "cp_min": -0.8456904849586655, "cp_max": 0.9999483166818915}, "tip": {"area": 0.7507977613442112, "normal_rms": 0.06799020596684653, "normal_max": 0.7894793350620118, "cp_min": -3.840503057489429, "cp_max": 0.9770630974763339}, "trailing": {"area": 0.33427319130688443, "normal_rms": 0.08347873174927951, "normal_max": 0.12484667330875678, "cp_min": 0.24071420756614315, "cp_max": 0.6190891578498766}}`.
  - component loads=`{"20": {"force_coefficient": [0.010736689292431928, 8.358332583952954e-06, 1.0721019097872827e-15], "cl": 1.0721019097872827e-15, "cd": 0.010736689292431928, "area": 4.949711709019088, "normal_rms": 0.0037280785748641486}, "30": {"force_coefficient": [-0.008468769373326161, -5.099701371428212e-05, -8.428648172348701e-16], "cl": -8.428648172348701e-16, "cd": -0.008468769373326161, "area": 7.982862331822595, "normal_rms": 0.03072990235076795}}`; component sum error=6.1658870040315236e-15 (pass=True).
  - raw mass by tag=`{"10": 0.0, "20": -0.0008444333250244276, "30": -0.06992986463123, "100": -0.003743190449633893, "201": 0.017125518702203427, "202": 0.017125518702222467}`; wake diagnostics=`0.015345496974444987` and `{"raw_paired_normal_mass_flux_jump_rms": 0.015345496974444987, "raw_paired_normal_mass_flux_jump_max": 0.25657437668280864, "raw_pressure_jump_rms": 8.987957108074585e-15, "wake_upper_normal_velocity_rms": 0.007672748487222507, "model": "Fixed planar wake with recorded tilt; no free-wake tangency or pressure-continuity iteration", "conservation": "Paired raw gradients, separate from weak nodal continuity"}`.
  - result=`results\wingbody_matrix_v1\t024_r035_a090\m0_a0\result.json`; analysis=`results\wingbody_matrix_v1\t024_r035_a090\m0_a0\analysis.json`.

- **m0_am5** (`M=0.0`, `alpha=-5.0°`, weak): status=`converged_subsonic_unvalidated_lifting`, converged=`True`, wall=56.56s, timed_out=False, timeout child terminated=False.
  - nodes/tets=113012/72426; CL pressure/Gamma=-0.37929984788248167/-0.3891329409453891; CD pressure=0.012698049148689827; raw Cp finite=276804/276804 (min/max=-4.077675654633793/0.9999562426537109).
  - wall RMS/max=0.030281268020443293/0.8158094725955953; global raw normal peak={'value': 0.8158094725955953, 'location': [0.002621593332195602, 1.999822049379482, 0.017828827331825385], 'region': 'tip'}; global Cp minimum={'value': -4.077675654633793, 'location': [0.0017151777891971094, 1.9998542224216567, 0.014492342342605696], 'region': 'tip'}.
  - junction/full-wall regions=`{"interior": {"area": 6.078498633125613, "normal_rms": 0.01477632261254857, "normal_max": 0.13000550248647968, "cp_min": -1.5368214834813472, "cp_max": 0.43340898705674524}, "junction": {"area": 0.883447139711817, "normal_rms": 0.01485828258331859, "normal_max": 0.19056863716479677, "cp_min": -1.903192548330094, "cp_max": 0.9988838370760048}, "leading": {"area": 4.885557315353157, "normal_rms": 0.010034415080401211, "normal_max": 0.18758197721074776, "cp_min": -1.6575245491616828, "cp_max": 0.9999562426537109}, "tip": {"area": 0.7507977613442112, "normal_rms": 0.09940937515225477, "normal_max": 0.8158094725955953, "cp_min": -4.077675654633793, "cp_max": 0.968232637885059}, "trailing": {"area": 0.33427319130688443, "normal_rms": 0.08517186451719973, "normal_max": 0.23765358696135566, "cp_min": -0.020615461743088037, "cp_max": 0.624435552420965}}`.
  - component loads=`{"20": {"force_coefficient": [0.009909234669812913, 8.900236713098338e-06, -0.06381429571718225], "cl": -0.06270781634820735, "cd": 0.015433309381364254, "area": 4.949711709019088, "normal_rms": 0.004219957694474721}, "30": {"force_coefficient": [-0.030317665398050527, -1.1987790725062563e-05, -0.31514890963536085], "cl": -0.3165920315342832, "cd": -0.0027352602326730265, "area": 7.982862331822595, "normal_rms": 0.03839875507006944}}`; component sum error=8.93729534823251e-15 (pass=True).
  - raw mass by tag=`{"10": 0.0, "20": -0.0008412200012823819, "30": -0.06966376038390179, "100": -0.0037289464798728744, "201": 0.5291604338876148, "202": -0.4950397320211719}`; wake diagnostics=`0.015287102725525378` and `{"raw_paired_normal_mass_flux_jump_rms": 0.015287102725525378, "raw_paired_normal_mass_flux_jump_max": 0.2555980337175987, "raw_pressure_jump_rms": 0.00750798395364639, "wake_upper_normal_velocity_rms": 0.061003666073631196, "model": "Fixed planar wake with recorded tilt; no free-wake tangency or pressure-continuity iteration", "conservation": "Paired raw gradients, separate from weak nodal continuity"}`.
  - result=`results\wingbody_matrix_v1\t024_r035_a090\m0_am5\result.json`; analysis=`results\wingbody_matrix_v1\t024_r035_a090\m0_am5\analysis.json`.

- **m02_a5** (`M=0.2`, `alpha=5.0°`, weak): status=`converged_subsonic_unvalidated_lifting`, converged=`True`, wall=143.77s, timed_out=False, timeout child terminated=False.
  - nodes/tets=113012/72426; CL pressure/Gamma=0.3840697350864504/0.39402581266216535; CD pressure=0.012971507650281378; raw Cp finite=276804/276804 (min/max=-4.114447911438048/1.009991856248581).
  - wall RMS/max=0.03034324364407023/0.8290953550900029; global raw normal peak={'value': 0.8290953550900029, 'location': [0.002621593332195602, 1.999822049379482, -0.017828827331825385], 'region': 'tip'}; global Cp minimum={'value': -4.114447911438048, 'location': [0.0017151777891971096, 1.9998542224216567, -0.014492342342605696], 'region': 'tip'}.
  - junction/full-wall regions=`{"interior": {"area": 6.078498633125613, "normal_rms": 0.014593635977469604, "normal_max": 0.12980849034512976, "cp_min": -1.5710325393295073, "cp_max": 0.4407178351052273}, "junction": {"area": 0.883447139711817, "normal_rms": 0.014767858667354432, "normal_max": 0.18679439592339248, "cp_min": -1.9352068113481946, "cp_max": 1.0087304529321683}, "leading": {"area": 4.885557315353157, "normal_rms": 0.00991905402790361, "normal_max": 0.1845143303331831, "cp_min": -1.6933048880435788, "cp_max": 1.009991856248581}, "tip": {"area": 0.7507977613442112, "normal_rms": 0.10026535826862723, "normal_max": 0.8290953550900029, "cp_min": -4.114447911438048, "cp_max": 0.9772718380456779}, "trailing": {"area": 0.33427319130688443, "normal_rms": 0.08458136101409731, "normal_max": 0.2384164738527741, "cp_min": -0.02251383509964328, "cp_max": 0.6325492422465566}}`.
  - component loads=`{"20": {"force_coefficient": [0.010222835462126503, 8.871216173835262e-06, 0.06475762171109198], "cl": 0.06362022059193201, "cd": 0.015827933105636797, "area": 4.949711709019088, "normal_rms": 0.004207841934304084}, "30": {"force_coefficient": [-0.030774571343015405, -1.4049817338915362e-05, 0.31898115346334743], "cl": 0.3204495144945174, "cd": -0.002856425455355336, "area": 7.982862331822595, "normal_rms": 0.03847875509285242}}`; component sum error=9.43689570931383e-16 (pass=True).
  - raw mass by tag=`{"10": 0.0, "20": -0.0008342137378720103, "30": -0.06764722895532993, "100": -0.0038158160621151183, "201": -0.4866967032776713, "202": 0.5204217871169634}`; wake diagnostics=`0.015271811063686084` and `{"raw_paired_normal_mass_flux_jump_rms": 0.015271811063686084, "raw_paired_normal_mass_flux_jump_max": 0.25652003264512113, "raw_pressure_jump_rms": 0.007649625734306171, "wake_upper_normal_velocity_rms": 0.061072883416166776, "model": "Fixed planar wake with recorded tilt; no free-wake tangency or pressure-continuity iteration", "conservation": "Paired raw gradients, separate from weak nodal continuity"}`.
  - result=`results\wingbody_matrix_v1\t024_r035_a090\m02_a5\result.json`; analysis=`results\wingbody_matrix_v1\t024_r035_a090\m02_a5\analysis.json`.

**固定网格门禁**
- alpha=0 误差 `{"cl_pressure": 5.936649911479111e-15, "cl_gamma": 2.9381880367747704e-16, "gamma_max": 4.5848343273236584e-15}`; passed=`True`.
- ±5° 对称误差 `{"cl_pressure": 6.827871601444713e-15, "cl_gamma": 1.2212453270876722e-15, "gamma_max": 1.3711254354120683e-14}`; passed=`True`.
- 所有变体 raw Cp 有限=`True`；组件力闭合=`True`；M=.2 状态=`converged_subsonic_unvalidated_lifting`.

### t036_r035_a090

- mesh SHA256 `9d01fb21a0388859822e80b60135d62098518e7f9378c8dc8c74f9f86afcb4e5`; dense minimum Jacobian `1.6770328036543972e-06`; mesh metadata `results/wingbody_matrix_v1/t036_r035_a090/mesh/mesh.json`.

- **m0_a5** (`M=0.0`, `alpha=5.0°`, weak): status=`converged_subsonic_unvalidated_lifting`, converged=`True`, wall=53.50s, timed_out=False, timeout child terminated=False.
  - nodes/tets=106684/69358; CL pressure/Gamma=0.39466637823090467/0.4087449362786466; CD pressure=0.01434980545472982; raw Cp finite=240084/240084 (min/max=-4.652809574556257/0.9999720584509547).
  - wall RMS/max=0.04076908394562185/1.1392262341803758; global raw normal peak={'value': 1.1392262341803758, 'location': [0.009750980332462258, -1.9997212160334095, -0.050496315009493975], 'region': 'tip'}; global Cp minimum={'value': -4.652809574556257, 'location': [6.882809274163958e-05, -1.9982844393551105, -0.004482720718402149], 'region': 'tip'}.
  - junction/full-wall regions=`{"interior": {"area": 6.1084827876944825, "normal_rms": 0.018746316214884956, "normal_max": 0.15594963461216232, "cp_min": -1.8990692460090743, "cp_max": 0.5241570078978407}, "junction": {"area": 0.8990415461102714, "normal_rms": 0.01763102623021141, "normal_max": 0.2123190692888602, "cp_min": -1.790289842482009, "cp_max": 0.9992990996973385}, "leading": {"area": 5.070579765990521, "normal_rms": 0.009925481481855838, "normal_max": 0.1063732189626706, "cp_min": -1.80262562469915, "cp_max": 0.9999720584509547}, "tip": {"area": 0.9395683241388457, "normal_rms": 0.12649448223106108, "normal_max": 1.1392262341803758, "cp_min": -4.652809574556257, "cp_max": 0.9239025687098253}, "trailing": {"area": 0.35719187279452, "normal_rms": 0.1093493221025935, "normal_max": 0.21303626094183947, "cp_min": 0.20345751313567373, "cp_max": 0.676664169496414}}`.
  - component loads=`{"20": {"force_coefficient": [0.012639588056722103, -1.673708626333191e-06, 0.06671048128013096], "cl": 0.06535501507330707, "cd": 0.01840569215319387, "area": 4.751410546324889, "normal_rms": 0.005041270423262308}, "30": {"force_coefficient": [-0.0327418292763139, -0.00045129204481214006, 0.3277047401812538], "cl": 0.3293113631575922, "cd": -0.00405588669846349, "area": 8.623453750403751, "normal_rms": 0.0506351720856916}}`; component sum error=5.329070518200751e-15 (pass=True).
  - raw mass by tag=`{"10": 0.0, "20": -0.0019188758839411785, "30": -0.1039634739992876, "100": 0.004318540872231935, "201": -0.4203022419923423, "202": 0.46440327234679535}`; wake diagnostics=`0.019687237334816678` and `{"raw_paired_normal_mass_flux_jump_rms": 0.019687237334816678, "raw_paired_normal_mass_flux_jump_max": 0.276847316486732, "raw_pressure_jump_rms": 0.011546509744518834, "wake_upper_normal_velocity_rms": 0.06235668333057317, "model": "Fixed planar wake with recorded tilt; no free-wake tangency or pressure-continuity iteration", "conservation": "Paired raw gradients, separate from weak nodal continuity"}`.
  - result=`results\wingbody_matrix_v1\t036_r035_a090\m0_a5\result.json`; analysis=`results\wingbody_matrix_v1\t036_r035_a090\m0_a5\analysis.json`.

### t024_r015_a090

- mesh SHA256 `c397f3ca65707af7dfd09fcc92c11d867afd26d35caac391f5d1ee5acd12af7a`; dense minimum Jacobian `2.540247891360765e-07`; mesh metadata `results/wingbody_matrix_v1/t024_r015_a090/mesh/mesh.json`.

- **m0_a5** (`M=0.0`, `alpha=5.0°`, weak): status=`converged_subsonic_unvalidated_lifting`, converged=`True`, wall=60.47s, timed_out=False, timeout child terminated=False.
  - nodes/tets=125639/78418; CL pressure/Gamma=0.36187540410815644/0.36991624949108026; CD pressure=0.012106735557883178; raw Cp finite=352080/352080 (min/max=-3.6417296382300846/0.9999777440949311).
  - wall RMS/max=0.03495536266836437/1.1950449820444735; global raw normal peak={'value': 1.1950449820444735, 'location': [0.019403501177973112, 1.9998029484539237, -0.04653128591299137], 'region': 'tip'}; global Cp minimum={'value': -3.6417296382300846, 'location': [0.009508537154485242, 2.0, -0.032886392627412], 'region': 'tip'}.
  - junction/full-wall regions=`{"interior": {"area": 6.262508790035396, "normal_rms": 0.01535211083674932, "normal_max": 0.12755568030311046, "cp_min": -1.4179816193501322, "cp_max": 0.37434648724771735}, "junction": {"area": 0.49177950647917035, "normal_rms": 0.015269067419035118, "normal_max": 0.13460299012172683, "cp_min": -1.4889659074393737, "cp_max": 0.9993708284629256}, "leading": {"area": 2.6022752342199884, "normal_rms": 0.014419682955198127, "normal_max": 0.2037398674127414, "cp_min": -1.5006385459774751, "cp_max": 0.9999777440949311}, "tip": {"area": 0.7526930902093046, "normal_rms": 0.10249830899079976, "normal_max": 1.1950449820444735, "cp_min": -3.6417296382300846, "cp_max": 0.9674160620924671}, "trailing": {"area": 0.3573148301732263, "normal_rms": 0.087719154939284, "normal_max": 0.2460932056334433, "cp_min": 0.024247403743397622, "cp_max": 0.5606123311822635}}`.
  - component loads=`{"20": {"force_coefficient": [0.0003409383781180974, -1.5945374233116414e-05, 0.019354468183790677], "cl": 0.019251103851501583, "cd": 0.0020264940547014424, "area": 1.7807723966920754, "normal_rms": 0.005505010808570365}, "30": {"force_coefficient": [-0.019819792231310012, 3.089580367515185e-05, 0.3421990622883596], "cl": 0.3426243002566549, "cd": 0.01008024150318312, "area": 8.685799054425011, "normal_rms": 0.03829066300817855}}`; component sum error=1.3808398868775384e-15 (pass=True).
  - raw mass by tag=`{"10": 0.0, "20": -0.000199945807079612, "30": -0.08730722562984099, "100": 0.001407568227932643, "201": -0.5092973004763832, "202": 0.5423380782193146}`; wake diagnostics=`0.014659166857586047` and `{"raw_paired_normal_mass_flux_jump_rms": 0.014659166857586047, "raw_paired_normal_mass_flux_jump_max": 0.29530388367671545, "raw_pressure_jump_rms": 0.006420252149742574, "wake_upper_normal_velocity_rms": 0.05969215687197232, "model": "Fixed planar wake with recorded tilt; no free-wake tangency or pressure-continuity iteration", "conservation": "Paired raw gradients, separate from weak nodal continuity"}`.
  - result=`results\wingbody_matrix_v1\t024_r015_a090\m0_a5\result.json`; analysis=`results\wingbody_matrix_v1\t024_r015_a090\m0_a5\analysis.json`.

### t024_r050_a090

- mesh SHA256 `b72d5794007efe8e85a8a26347c9c0c67ccc7e8f494d9d3af28b69286f7e59fa`; dense minimum Jacobian `2.2070954906663974e-07`; mesh metadata `results/wingbody_matrix_v1/t024_r050_a090/mesh/mesh.json`.

- **m0_a5** (`M=0.0`, `alpha=5.0°`, weak): status=`converged_subsonic_unvalidated_lifting`, converged=`True`, wall=48.91s, timed_out=False, timeout child terminated=False.
  - nodes/tets=106232/69050; CL pressure/Gamma=0.3947884439152032/0.4076600473548; CD pressure=0.01335322817266225; raw Cp finite=238500/238500 (min/max=-3.664519835009484/0.9999863613312581).
  - wall RMS/max=0.0301402892705028/0.8830223431759525; global raw normal peak={'value': -0.8830223431759525, 'location': [0.02870614810176461, -2.0, -0.05542243592568838], 'region': 'tip'}; global Cp minimum={'value': -3.664519835009484, 'location': [0.007576713330031963, 2.0, -0.029454697325006607], 'region': 'tip'}.
  - junction/full-wall regions=`{"interior": {"area": 6.246735453863853, "normal_rms": 0.015160215175203105, "normal_max": 0.10913178321678424, "cp_min": -1.7618175363626292, "cp_max": 0.5145485854619434}, "junction": {"area": 0.9528960235294109, "normal_rms": 0.0153033671150094, "normal_max": 0.12456956082180956, "cp_min": -2.2009087508096874, "cp_max": 0.99781653071249}, "leading": {"area": 6.6548281242661345, "normal_rms": 0.008774840939099821, "normal_max": 0.20511674847763423, "cp_min": -1.8901401284253718, "cp_max": 0.9999863613312581}, "tip": {"area": 0.7556185845720047, "normal_rms": 0.1073279381372039, "normal_max": 0.8830223431759525, "cp_min": -3.664519835009484, "cp_max": 0.9689247296024983}, "trailing": {"area": 0.3287503902853942, "normal_rms": 0.09055008423988282, "normal_max": 0.20128099559737306, "cp_min": -0.20503097301993845, "cp_max": 0.7515487584013321}}`.
  - component loads=`{"20": {"force_coefficient": [0.017795709392200494, -4.9655864704644635e-06, 0.09161168265432251], "cl": 0.08971207427370091, "cd": 0.025712475591391844, "area": 7.363441975602267, "normal_rms": 0.005060937863777325}, "30": {"force_coefficient": [-0.038901374341807494, -0.0002548307328395635, 0.3028382825613475], "cl": 0.3050763696414892, "cd": -0.012359247418731477, "area": 7.5753866009145305, "normal_rms": 0.04203048037707615}}`; component sum error=1.3156142841808105e-14 (pass=True).
  - raw mass by tag=`{"10": 0.0, "20": -0.0009021868150307522, "30": -0.07336938227108206, "100": -0.003892260045919471, "201": -0.41044665948186754, "202": 0.4417731940195303}`; wake diagnostics=`0.014766042173556064` and `{"raw_paired_normal_mass_flux_jump_rms": 0.014766042173556064, "raw_paired_normal_mass_flux_jump_max": 0.28979663464035493, "raw_pressure_jump_rms": 0.009231140473385654, "wake_upper_normal_velocity_rms": 0.0631160904249521, "model": "Fixed planar wake with recorded tilt; no free-wake tangency or pressure-continuity iteration", "conservation": "Paired raw gradients, separate from weak nodal continuity"}`.
  - result=`results\wingbody_matrix_v1\t024_r050_a090\m0_a5\result.json`; analysis=`results\wingbody_matrix_v1\t024_r050_a090\m0_a5\analysis.json`.

### t024_r035_a099

- mesh SHA256 `79c94fd5924290912e7f8a5887e8a319dee2d84938ccbc68f8b13e4e5be9997f`; dense minimum Jacobian `6.494585017525149e-07`; mesh metadata `results/wingbody_matrix_v1/t024_r035_a099/mesh/mesh.json`.

- **m0_a5** (`M=0.0`, `alpha=5.0°`, weak): status=`converged_subsonic_unvalidated_lifting`, converged=`True`, wall=52.82s, timed_out=False, timeout child terminated=False.
  - nodes/tets=114059/73104; CL pressure/Gamma=0.383291279029758/0.39393078193949527; CD pressure=0.013494199717333216; raw Cp finite=279000/279000 (min/max=-3.8935926126754836/0.9999672364945209).
  - wall RMS/max=0.033910646883164544/0.8229702761732189; global raw normal peak={'value': 0.8229702761732189, 'location': [0.007283382949843113, -1.9998055468705287, -0.02928986757965844], 'region': 'tip'}; global Cp minimum={'value': -3.8935926126754836, 'location': [0.009015753252524397, -2.0, -0.03205023463537449], 'region': 'tip'}.
  - junction/full-wall regions=`{"interior": {"area": 6.116993694600957, "normal_rms": 0.015578501861304425, "normal_max": 0.17099422550320975, "cp_min": -1.5616357017982474, "cp_max": 0.4615408419117907}, "junction": {"area": 0.9196296416139026, "normal_rms": 0.012844416300119807, "normal_max": 0.12394077014266369, "cp_min": -1.8749420669485284, "cp_max": 0.999117607165758}, "leading": {"area": 4.699810742069035, "normal_rms": 0.010230105705782259, "normal_max": 0.1679079253478648, "cp_min": -1.6782250504751048, "cp_max": 0.9999672364945209}, "tip": {"area": 0.757913272595085, "normal_rms": 0.11254663237558107, "normal_max": 0.8229702761732189, "cp_min": -3.8935926126754836, "cp_max": 0.9711393754145103}, "trailing": {"area": 0.31429802970658455, "normal_rms": 0.09770978838842076, "normal_max": 0.3698336428545381, "cp_min": -0.3176049514662955, "cp_max": 0.6102281343525923}}`.
  - component loads=`{"20": {"force_coefficient": [0.0075403290877925026, -2.960025074709375e-07, 0.06622933846905395], "cl": 0.0653201328587869, "cd": 0.013283903045082306, "area": 4.954087452960997, "normal_rms": 0.003914946573572447}, "30": {"force_coefficient": [-0.027503514986933108, 2.6854282326086586e-05, 0.31677949852435294], "cl": 0.3179711461709798, "cd": 0.00021029667225103826, "area": 7.854557927624566, "normal_rms": 0.04319211625762503}}`; component sum error=8.659739592076221e-15 (pass=True).
  - raw mass by tag=`{"10": 0.0, "20": -0.0009194373046937478, "30": -0.06991831997949911, "100": -0.002173181184039308, "201": -0.4349901487563127, "202": 0.4681193228539409}`; wake diagnostics=`0.01569086788558782` and `{"raw_paired_normal_mass_flux_jump_rms": 0.01569086788558782, "raw_paired_normal_mass_flux_jump_max": 0.26970194451986496, "raw_pressure_jump_rms": 0.008767540734943305, "wake_upper_normal_velocity_rms": 0.06926317990904945, "model": "Fixed planar wake with recorded tilt; no free-wake tangency or pressure-continuity iteration", "conservation": "Paired raw gradients, separate from weak nodal continuity"}`.
  - result=`results\wingbody_matrix_v1\t024_r035_a099\m0_a5\result.json`; analysis=`results\wingbody_matrix_v1\t024_r035_a099\m0_a5\analysis.json`.

## 解释边界

求解器返回收敛只说明离散非线性方程达到自身残差条件。组件载荷闭合、有限 Cp、壁面无穿透、wake raw 通量及压力跳跃是不同诊断。报告保留完整 wall 区域、全域峰值和原始 mass flux；不通过裁剪极值或几何处理美化结果。冻结尾迹仍是模型限制；本矩阵只评估指定网格/几何下的鲁棒性，不能单独证明厚翼身 Cp 或整体气动精度。

Runner SHA256 `eedfc3fe51c0b911a997f91693c596f61366e29d2a77b6a17287dce0ca58bd25`；总经过时间 668.42 s。
