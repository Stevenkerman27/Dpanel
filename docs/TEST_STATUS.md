# 测试状态（2026-11-03）

本页记录**当前工作树实测**的测试基线。历史 PR 的收口报告在
[archive/notes/TEST_HEALTH_2026-10-30.md](archive/notes/TEST_HEALTH_2026-10-30.md)（master `91d4d2a`、NumPy 链口径），
两边的数字不能直接比较：分支不同、依赖链不同（NumPy vs torch）。

## 1. 环境

Python 3.11.17（`.conda_env`）、pytest 9.1.1。
可用：numpy、scipy、jax、meshio；**不可用：torch、gmsh、numba、vsp（OpenVSP）**。

## 2. 实测（2026-11-03，提交 cb5f771）

整仓 `pytest test -q` 在收集阶段中断：7 个 torch 测试文件
`ModuleNotFoundError: No module named 'torch'`，因此整仓数字不可用。

排除这 7 个文件后：**613 passed, 66 failed, 52 skipped**（116.5 s）。

| 失败原因 | 数量 | 说明 |
| --- | --- | --- |
| 缺 torch | 62 | P1 solve chain 现在要求 torch；`docs/issue92/references/` 的 NumPy 冻结答案只用于回放，不在本机重算 |
| 沙箱禁止子进程管道 | 4 | CLI 用例走 `subprocess`，在 DSH 沙箱下 `PermissionError: WinError 5`，换正常终端即消失 |
| 文档关键词 | 0（已修） | `docs/ARCHITECTURE_WAKE_MODELS.md` 中文化后丢了测试要求的英文关键词，补术语对照附录后该用例通过，上面的 66 个失败已不含它 |

## 3. 复现

```bash
# 整仓（会因缺 torch 中断）
./.conda_env/python.exe -m pytest test -q

# 非 torch 子集：就是上表 613/66/52 的来源
./.conda_env/python.exe -m pytest test -q \
  --ignore=test/test_torch_half_checkpoint.py --ignore=test/test_torch_half_core_cpu.py \
  --ignore=test/test_torch_half_lifting.py --ignore=test/test_torch_panel_kernels.py \
  --ignore=test/test_torch_panel_nonlifting.py --ignore=test/test_torch_panel_symmetry.py \
  --ignore=test/test_torch_panel_validation.py
```

## 4. 未验证（不得声称通过）

- torch 链：上述 7 个文件加 62 个 P1 用例，本机无 torch，**未验证**。
- 历史记录里的“827 passed”口径来自 research 分支 + torch 环境，本机没有复现。
- 物理精度仍未验收（`physical_accuracy_accepted=false`），测试通过不等于精度通过。
