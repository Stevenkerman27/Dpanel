# B3 尾迹位置敏感性证据（issue #113）

实验结果与方法解释见 `docs/ARCHITECTURE_WAKE_MODELS.md` 第 10 节和
`STATUS.md`。`manifest.json` 的 `execution_complete` 只表示计划作业和
独立审计已收集，不表示物理精度或所有状态通过验收。

本目录提交四份 E1–E4 汇总、三组冻结初始输入和每例原始数值/审计 JSON、
干预历史、初始网格、队列与收集器回执、日志以及图。一次失败的
`elliptic/e3_freeze_1_serialization_failure` 尝试和早期几何审计错误日志
均保留。`manifest.json` 还记载了当时本机逐轮 `iteration_*.npz` 的路径和
SHA-256；为避免重复保存大量快照，这些逐轮文件没有提交 Git，仍保留在
本机实验目录。因此不能仅凭仓库克隆直接重做逐轮几何审计；摘要和审计
结果本身可直接读取，需重算时应从冻结输入和对应代码重新生成快照。
目录内 `.gitattributes` 禁止 JSON 和日志的换行转换，以保留记录的原始哈希。

原实验是先分阶段启动后执行 `--collect-only`，并非一次运行下列新目录
完整编排。该新目录入口已提供，但本轮未端到端复跑：

```powershell
$env:OPENBLAS_NUM_THREADS=1; $env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1
python -m tools.wake_sensitivity_run --out runs/verification/wake_sensitivity_new
```

历史数值作业的实际命令、参数、原始代码哈希及退出情况分别记录在每例
manifest/history、`queue.json`、`driver_receipt.json` 和日志中。初始求解与
干预所用脚本有当时尚未提交的改动，应按每个文件的源哈希核对，不把
实验起点 commit 误称为运行时整个工作树均已提交。当前脚本的完整新目录
运行可能随运行环境或数值轨迹产生差异，不能承诺逐位复现。

本轮相关回归在 Windows 11、项目 myml Python 3.13.14 环境中运行：
八个针对性文件 `76 passed, 1 xfailed`；全量 `python -m pytest -q test
-p no:cacheprovider` 为 `11 failed, 815 passed, 21 skipped, 1 xfailed`，
失败 ID 与已有基线一致。完整输出见 `merge_review_tests.log` 与
`merge_full_pytest.log`。预期失败是已知的均匀轴向流不动点缺陷，不是通过。
