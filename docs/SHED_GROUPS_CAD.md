# 真实 CAD：两片独立尾迹（Step B 第二批）

本文件记录 issue #41 第 2/3/4 条的第二批交付：在**真实 OCC 半域**里生成两片
互相独立的水平尾迹（z = 0 与 z = 0.24），并驱动 v2 多组求解器求解。
第一批（装配、校验、入口、小夹具）见 `docs/SHED_GROUPS_STEPB.md`。

## 1. 不改动的东西

v1 入口把尾迹角点 z 硬编码为 0（`panel/cad_half.py:179-189`），`build_half_surface`
严格只接受一片水平尾迹。本轮**没有改动** `cad_half.py`、`half_native.py`、
`half_surface_mesh.py`、`kernel.py`、`pressure_stencil.py`，也没有改默认 CLI 与默认模式。

新增文件：

| 文件 | 作用 |
| --- | --- |
| `panel/wake_group_declaration.py` | 与 gmsh 无关的声明解析：站位去重、TE 限制到 y>=0、z 平面唯一性 |
| `panel/cad_half_groups.py` | 多尾迹 OCC 半域构建：逐组拉升尾迹片、fragment、按 z 分回各组、写 surface 与 manifest |
| `panel/half_group_surface.py` | 多片尾迹的半域装配：每片各自求 TE 边、上下 bank、站位与自由度 |
| `cases/panel_half/two_layer_wing/` | 两层层流翼 STEP 夹具（含生成器与声明） |
| `test/test_wake_group_declaration.py` | 9 passed，不需要 gmsh |
| `test/test_cad_half_groups_wake_sheet.py` | 2 passed，gmsh 门控，真实 STEP |

## 2. 夹具几何

两层层流翼（双翼式），两层都在 y = 0 对称、无垂尾、无机身：

- 半展 1.6、根弦 1.0、梢弦 0.5、NACA 2412 剖面、后掠 0、下沉 z = 0 与 z = 0.24；
- 两个组 `lower_wing` / `upper_wing`，`root_condition` 都是 `symmetry_plane`；
- 每组的 TE 多段线由生成 STEP 的同一公式导出（不是从网格反推）。

没有用 `conv`（常规布局）案例：它的 T 尾根部埋在垂尾盒里，平尾根 TE ≈ 3.48 与垂尾
TE ≈ 3.49 几乎重合，第二片水平尾迹必然与垂尾相交，违反 D9/D10。要干净地满足契约，
需要**没有垂尾**的两层升力面几何。

## 3. 构建流程

1. 归一化：`importShapes` 后 `translate(-origin)`、`dilate(1/chord)`，与 v1 一致；
2. 逐组逐站拉尾迹条带矩形：角点 z 用该组的**归一化 z**（不是 0），下游 x = radius；
3. `occ.fragment(fluid, wake_inputs)` 后靠 `getEmbedded(3, volume)` 与 `getAdjacencies(2, tag)`
   让每片尾迹三角形**不重复**地归属到一个组；
4. 两侧都做两遍 attempt（先试贴体，再放宽），失败重试；
5. 半域装配后按三角形质心 z 与各组的归一化 z 匹配（容差 1e-7）把尾迹三角形分回各组；
6. 每组的上下 bank 由 **wake 与 wall 的切割边**决定：两条 bank 面对该组 z 平面的质心 z，
   大者为 upper、小者为 lower（因此对抬高的尾迹同样成立）；
7. 站位自由度：root 自由、tip 为 -1（`half_contact_station_map`）。

## 4. 声明与 manifest

声明 schema `dpanel.half_wake_groups_declaration.v1`，每组给出 `group_id`、`points`（全翼展 TE，
物理 SI 米）、`z`（物理米），可选 `role` / `kind` / `root_condition`。构建器输出的 `groups.json`
换成求解器要的 v2：`wake_faces` 是**尾迹三角形的索引**（`surface.json` 的 `wake_faces` 是全池），
并带 `shed_edges` / `upper_te` / `lower_te` / `stations` / `station_dof` / `wake_z`。

`manifest.json` 报告 `right_wall_panels`、`right_wake_panels`、逐组 `wake_panels` /
`shed_edges` / `stations` / `circulation_unknowns`（= 自由度个数，即 `max(dof)+1`）/ `wake_z`，
以及 `free-stream` 来源与耗时；`physical_accuracy_accepted` 恒为 false。

## 5. 两档实测

`--wake-strips 8`（两组同 8 个自由度）、`--alpha-deg 5`、Sref 4.8（全机双侧）、c = 1.0：

| size | 壁面 | 尾迹 | 未知量 | CL | CD_p | CM_y | Cp_min | 真残差 | 热解 s | 峰值 MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.20 | 508 | 368 | 524 | 0.285245 | -0.012540 | -0.049386 | -3.280 | 1.07e-15 | 2.48 | 1151.3 |
| 0.13 | 896 | 528 | 912 | 0.298993 | -0.011177 | -0.046101 | -4.243 | 1.31e-15 | 2.55 | 1178.0 |

逐组读数（`gamma_sum` / `te_jump_rms` / Kutta 残差）：

| size | lower gamma_sum | upper gamma_sum | lower TE 跳跃 | upper TE 跳跃 |
| --- | --- | --- | --- | --- |
| 0.20 | +1.91209 | +0.017534 | 0.233255 | 0.025445 |
| 0.13 | +2.051714 | -0.095161 | 0.250312 | 0.029068 |

两组 Kutta 行残差 6.8e-17 / 2.7e-17（0.20）与 9.5e-17 / 6.6e-17（0.13）；块宽度 `[8, 8]`
与 manifest 的 `circulation_unknowns` 一致；未知量 = 壁面 + 两组自由度。

可微性：前向 AD 的 `dCL/dalpha` = 1.9025 /rad（0.0332054 /deg）与 2.2112 /rad（0.0385930 /deg），
中心差分 h = 1e-4 / 1e-5 的相对差 1.05e-8 / 1.02e-10 与 9.92e-9 / 7.46e-11。

## 6. 发现（只报告，不作精度结论）

1. **两档之间 CL 变化 4.8%**（0.285245 -> 0.298993），表面上进 5% 带；但**同一 size 重跑**
   （0.13，同一代码同一输入）得到 928 壁面 / CL 0.305645，与本次 896 壁面 / 0.298993 差 2.2%。
   也就是说 5% 的门槛已经与网格本身的可重复性同量级，**这里不能声明收敛**。
2. **近场 CD_p 在两种分辨率下都是负的**（-0.012540 / -0.011177，随加密绝对值减小）。
   这不是推力，是近场压力积分在正负相消下的表现；与 issue #42 那批椭圆翼序列方向一致，
   再次说明近场 CD 不是这一离散的良定义泛函。
3. **上层翼的环量在两档之间换号**（+0.0175 -> -0.0952）。叠放对不是对称对，上层翼处在
   下层翼的诱导速度场里，粗网格下几乎为零、细网格下明显为负。任何"尾翼是否承载升力"的
   陈述都必须绑定一个已收敛的网格，本轮只给出测量值。
4. 下层翼 `gamma_sum` 从 1.912 涨到 2.052（+7.3%）：它不是守恒量（TE 跳跃到站位的插值、
   尾迹三角形 184 -> 264 都参与），不能当收敛指标。
5. 数值层面全部干净：真残差 ~1e-15、Kutta 行 ~1e-16、镜像算符重建差 2.8e-17 / 5.6e-17、
   尾迹算子相对峰值差 0.0。**这个算例证明的是"两片独立尾迹能装进同一个线性系统并解出来"，
   不是"双翼气动系数已经收敛"。**

## 7. 测试

`test/test_cad_half_groups_wake_sheet.py`（gmsh 门控，2 passed）：

- 两片尾迹的三角形索引**互不相交**且并集等于 `surface.json` 的全部尾迹三角形；
- 上层尾迹所有节点 z 恒为 0.24；
- 求解后块宽度等于 manifest 的 `circulation_unknowns`，未知量 = 壁面 + 两组自由度之和；
- 真残差 < 1e-10、Kutta 行 < 1e-12；
- 两片尾迹的 surface 上 `v1_equivalence` 与 `full_oracle` 都必须 `performed=False`，
  不能悄悄只取第一片；
- 空声明被构建器拒绝，写出 `failure.json` 且不产出 `surface.json`。

全套（contract + entry + declaration + CAD）合计 **37 passed / 20.6 s**。

## 8. 限制

- 没有多尾迹的全模 oracle：目前没有两片尾迹的全机模型，`full_oracle` 如实 `performed=False`；
- 稠密 LU，没有矩阵无关路径；
- 只有常数 z 的水平尾迹，没有倾斜尾迹/侧滑/垂尾涡；
- 没有场级双侧指标（尾迹压力跳跃的通量审计仍要复用 PR #30 的模块）；
- Gmsh 不可位复现：同一 size 的壁面数会在 8xx 内浮动，见发现 1。

## 9. 复现命令

```text
python -m panel.cad_half_groups --wing-cad cases/panel_half/two_layer_wing/two_layer.step \
  --wake-groups cases/panel_half/two_layer_wing/groups.json --out runs/mesh_s020 \
  --chord 1.0 --reference-area 4.8 --size 0.2 --far-size 1.5 --radius 5 \
  --wake-strips 8 --alpha-deg 5 --ncpu 8

python -m panel.run_half_groups --surface runs/mesh_s020/surface.json \
  --groups runs/mesh_s020/groups.json --out runs/solve_s020.json --max-panels 4000
```

环境：Python 3.11.17、NumPy 2.4.6、JAX 0.10.2、gmsh 4.15.2（`audit/runtime/python`）、
Windows 10.0.26200；`PYTHONPATH` 需同时包含本工作树与 `audit/runtime/python`。
