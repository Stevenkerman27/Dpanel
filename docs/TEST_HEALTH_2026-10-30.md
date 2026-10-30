# master 测试健康度收口（2026-10-30）

基线 master `91d4d2a`：`11 failed, 640 passed, 9 skipped`（整仓 `pytest test`，Py 3.11.17 / NumPy 2.4.6 / pytest 9.1.1）。
本 PR 之后：`648 passed, 12 skipped`，0 failed。

三类失败、三种处置：**两处在生产代码里是错的判据**（改代码），**七处是测试侧过期**（改测试），**三个是装包缺数据**（改为显式 skip，不掩盖）。

## 1. `panel/matrix_free.py` `mirror_gate`：TE 配对判据与 XZ 对称矛盾

旧代码（`:298-300`）：

~~~python
te_mirror = pair[op.upper]
lower_set = set(op.lower.tolist())
te_paired = int(sum(1 for j in te_mirror if j in lower_set))
~~~

要求 upper 后缘边的镜像落在 **lower** 集合。XZ 对称面是 y→−y、**z 不变**，upper（z>0）的镜像仍是 upper。用 `test/test_matrix_free_symmetry_gate.py` 的夹具（`pair=[1,0,3,2]`、`upper=[0,1]`、`lower=[2,3]`）复算：`pair[upper]=[1,0]⊄{2,3}` → `te_paired=0` → 任何真正 y 对称的网格都被判 `blocked`，半模 RAE 降阶被误拦。

现改为 bank 一致（upper 与 upper 集合求交、lower 与 lower 集合求交，取 min），并单独报 `te_edges_paired_upper` / `te_edges_paired_lower`。同样的论证以节点级写在 `panel/symmetry.py` 的 `te_bank_gate` 文档串里（issue #18 的独立审计结论），现在两处一致。

## 2. `panel/matrix_free.py` `mirror_gate`：贴在对称面上的面元没有原因码

`straddling` 只统计 **|y|>atol** 的自映面，真正躺在 y=0 上的自映面既不进 `straddling` 也不产生 reason（`test_on_plane_self_nearest_wall_panel_is_not_half_reduced` 断言 reasons 里含 `self-mirror` → 失败）。判定本身正确（这类面元确实改变未知量计数），缺的是诊断。现新增 `self_panels_on_plane = self_panels - straddling` 与对应 reason。

## 3. `test/test_p1_coupled_material_wake.py`：6 个用例的 mock 签名过期

生产侧 `panel/p1_coupled_material_wake.py:430` 调用 `TotalVelocity(current, state, method=velocity_method)`，真实签名 `:55` 为 `def __init__(self, mesh, state, *, method="analytic_edges")`。测试的 mock 不接受该关键字：

- `:317` `lambda model, state: field` → `lambda model, state, **kw: field`
- `:243` `class SmoothField(UniformField)` 的 `__init__(self, model, state)` → 接受 `method`
- `UniformField` 增加类属性 `method = 'analytic_edges'`（生产侧 `:436` 会把 `field.method` 记进 `sampling_kernel`）

## 4. VortexAD 三个测试：缺 package data → skip（不是 fail）

pip 安装的 `VortexAD` 在 `core/geometry/sample_meshes/pm/` 只有一个 2 字节的 `__init__.py`，STL 一个都没有，于是三个测试都停在 `meshio._exceptions.ReadError: ... not found`。这不是代码或物理问题：

- 源码 checkout `runs/backend_probes/src/VortexAD/VortexAD/core/geometry/sample_meshes/pm/` 里 8 个 STL 齐全（`naca0012_LE_TE_cluster.stl` 475 505 B 等）；
- 包根前置 `PYTHONPATH=runs/backend_probes/src/VortexAD` 时三个文件 **3 passed in 177.21 s，exit 0**（注意必须是 `src/VortexAD` 这一层，`src` 不是包根）。

现改为显式前置检查 + `pytest.skip`，跳过原因里写清缺哪个文件、怎么修。**不静默**：有数据时必须真跑。

## 验收命令

~~~bash
# 整仓
python -m pytest test -q -p dsh_tmp_plugin -p no:cacheprovider
#   -> 648 passed, 12 skipped in 152.12s

# 本次触达的文件
python -m pytest test/test_matrix_free_symmetry_gate.py test/test_p1_coupled_material_wake.py \
    test/test_symmetry_diagnostics.py test/test_vortexad_backend.py \
    test/test_vortexad_mesh_vjp.py test/test_vortexad_raw_cp.py -q -rs
#   -> 42 passed, 3 skipped in 2.32s
~~~
