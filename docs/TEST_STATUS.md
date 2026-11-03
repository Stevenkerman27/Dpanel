# 测试状态（2026-11-03）

本页记录**当前工作树实测**的测试基线。历史 PR 的收口报告在
[archive/notes/TEST_HEALTH_2026-10-30.md](archive/notes/TEST_HEALTH_2026-10-30.md)（master `91d4d2a`、NumPy 链口径），
两边的数字不能直接比较：分支不同、依赖链不同（NumPy vs torch）。

## 1. 环境

Python 3.11.17（`.conda_env`）、pytest 9.1.1。
可用：numpy、scipy、jax、meshio；**不可用：torch、gmsh、numba、vsp（OpenVSP）**。

## 2. 实测

### 2.1 重组后（2026-11-03，当前工作树）

```bash
./.conda_env/python.exe -m pytest test -q --ignore-glob='test/*torch*.py' \
  --ignore=test/test_p1_coupled_material_wake.py \
  --ignore=test/test_p1_lifting_g2_pressure_floor.py \
  --ignore=test/test_p1_geometry_contact.py \
  --ignore=test/test_p1_gate_contact.py
```

**564 passed, 16 failed, 15 skipped**（115.4 s）。

16 个失败全部是同一条环境原因 `ValueError: solve_backend must be None or 'torch'`
（P1 链只保留 torch 实现）：

| 文件 | 失败数 |
| --- | ---: |
| test/test_p1_lifting_g1.py | 4 |
| test/test_p1_lifting_g2.py | 4 |
| test/test_gate_mode.py | 2 |
| test/test_p1_steady_free_wake.py | 2 |
| test/test_p1_steady_free_wake_guards.py | 2 |
| test/test_analytic_batch.py | 1 |
| test/test_wake_step_scale.py | 1 |

本轮同时把 `runs/` 按功能重组（见 [../runs/README.md](../runs/README.md)），
这次实测没有任何 `No such file` / 路径类失败，说明重组没有改坏测试。

### 2.2 重组前基线（提交 `86fab48`，只排除 7 个 torch 文件）

613 passed, 66 failed, 52 skipped（116.5 s）。当时的失败构成是 62 个缺 torch +
4 个沙箱禁止子进程管道（CLI 用例的 `PermissionError: WinError 5`，换普通终端即消失）。
子进程类失败在本轮改用 `--ignore-glob` 的口径里不再出现。

## 3. 复现

```bash
# 整仓（会在收集阶段因缺 torch 中断，仅用于确认环境）
./.conda_env/python.exe -m pytest test -q

# 非 torch 子集：上表 2.1 的来源
./.conda_env/python.exe -m pytest test -q --ignore-glob='test/*torch*.py' \
  --ignore=test/test_p1_coupled_material_wake.py \
  --ignore=test/test_p1_lifting_g2_pressure_floor.py \
  --ignore=test/test_p1_geometry_contact.py \
  --ignore=test/test_p1_gate_contact.py
```

## 4. 未验证（不得声称通过）

- torch 链：17 个 `test/*torch*.py` 文件与 16 个 `solve_backend` 用例，本机无 torch，**未验证**。
- 历史记录里的“827 passed”口径来自 research 分支 + torch 环境，本机没有复现。
- 物理精度仍未验收（`physical_accuracy_accepted=false`），测试通过不等于精度通过。
