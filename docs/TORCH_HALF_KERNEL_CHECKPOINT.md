# PyTorch 半模解析核：按目标面元分块重计算（可选性能试验）

> **范围**：堆叠在已经本地真实 RTX 5060 CPU/CUDA 通过的 PR #23 上，
> 只改变自动微分中间张量的保存策略。半模源/偶极势、镜像、尾迹一阶矩、
> Kutta、压力模型、LU 与显式伴随 **全部保持不变**。
> `physical_accuracy_accepted=false`，不调整生产求解入口或 GRASS 训练。

## 为什么单独试验

本地 Torch 2.13.0+cu132 / RTX 5060 真实测量：

| 半模未知量 | PyTorch CUDA 热前向 | 反向 | CUDA peak allocated |
|---|---:|---:|---:|
| 418 | 0.1097 s | 0.5241 s | 371.5 MB |
| 827 | 0.2215 s | 1.0579 s | 1,151.9 MB |

这里 `allocated` 包含测量期的 PyTorch 图、几何、AIC 及分解等显存，
并不等价于纯矩阵占用。两个规模不足以推出线性内存复杂度；**最终半模
矩阵与 LU 仍是 O(N²) 存储和 O(N³) 分解**。CUDA `first` 约 15.5s
是设备/初次编译开销，冷热求解与反向时间应分开报告。

当前 `torch_panel.half_lifting` 的解析核每次对全体源面元与一小块目标面元
生成临时半径、叉积、角度、边线积分等张量。在完整载荷对坐标进行 VJP 时，
PyTorch 保存这些中间量以执行反向，占用随 pair count 增长。
这一试验只在每个**目标块**调用
`torch.utils.checkpoint.checkpoint(...,use_reentrant=False)`：
保留源势/偶极/尾迹矩阵数值，反向重新计算块内临时表达式而不是从前向
保留它们。没有混合精度、epsilon 核软化、舍弃梯度或自迹改变。

## 接口和正确性

`triangle_potential_torch(..., checkpoint_blocks=False)` 和
`solve_half_torch(..., checkpoint_kernel=False)` 默认都是 **False**，
这是保证 PR23 本地已接受路径完全不变的关键。显式设置 True 后
会对真实机身面、镜像面和两套 wake 的源/偶极积分统一重计算。
如果输入不需要梯度或执行 `torch.no_grad()`，不会创建额外检查点。

**自迹与分块索引**：self 双层迹必须是 `-1/2`，
反向重计算不得从目标循环的末尾拿错 `target_panel_ids`。
当前实现将每个目标块的真实行 ID 绑定为局部函数的不可变默认参数，
保持分块偏移一致。压力恢复使用已经冻结的切边感知邻域，不在核重计算里
再次选择面元邻居。

`test/test_torch_half_checkpoint.py` 对 CPU 比较 checkpoint 开/关的半模
A/RHS/μ/γ/`Cp_raw_half`/力矩和包含局部压力的顶点、站位、来流梯度；
额外测试多块矩形自迹和 Torch-only 解析核的 VJP。
如有真实 CUDA 卡，增加 CUDA 开关前后梯度等价性测试。

数学上这项改变不应超过 Float64 舍入误差；
**性能上 checkpoint 可能变慢或不节省显存**。若实测峰值无改善，
应报告负结果，不必把它设为默认，也不能因此放宽梯度容差。

## 本地 A/B：在完全相同的真实半模网格上分别新进程测量

本地保存的两个来源（无需重跑 Gmsh/OCC）：

- `runs/half_cad_wingbody_v1/surface.json`：396 wall、22 gamma，
  半模 418 未知量；
- `runs/half_cad_wingbody_2x/surface.json`：805 wall、22 gamma，
  半模 827 未知量。

先单测：
```powershell
git fetch origin
git switch perf/torch-half-kernel-checkpoint
python -m pytest -q test/test_torch_half_core_cpu.py test/test_torch_half_lifting.py test/test_torch_half_checkpoint.py
```

对每个真实输入（此处 `<ABS_SURFACE>` 应指向以往 run 的**绝对路径**）：

```powershell
# 独立进程 A：沿用已验收的默认解析核
python -m panel.verify_torch_half --surface <ABS_SURFACE> `
  --out runs/torch_ckpt_off_418_v1.json --device cuda `
  --block 16 --derivatives

# 独立进程 B：仅改变 kernel activation checkpoint
python -m panel.verify_torch_half --surface <ABS_SURFACE> `
  --out runs/torch_ckpt_on_418_v1.json --device cuda `
  --block 16 --derivatives --checkpoint-kernel
```

805 wall 重复另存 `*_827_v1.json` 文件名。报告两种模式的：

- `checkpoint_kernel`、`compared_fields/passed_fields`、23 个
  逐字段误差与 `torch_true_relative_residual`；
- `geometry_station_vjp` 中 Torch/JAX 三步长中心差分；
- `torch_first_seconds`、`torch_hot_seconds`、
  `torch_backward_seconds`；
- `cuda_peak_allocated_bytes` 和 `cuda_peak_reserved_bytes`，
  并单独记录采样进程 RSS。**不要把同一进程两种配置复用的
  CUDA allocator 峰值当 A/B 结果**。

判读原则：字段/梯度全通过只是“计算等价”；记录 checkpoint 开关的
实际显存减少百分比、反向变慢比例、GPU 型号和版本后，才决定是否
选择默认值。没有 CUDA 环境时执行 CPU 等价即可，CUDA 结论写“未测”。

## 不包括

没有实现 Torch GMRES/FMM/H²、动态 CAD 拓扑梯度、非平面 wake、
独立压力物理精度基准或 GRASS 训练。即使峰值内存下降，
最终 dense `AIC+LU` 存储仍会限制数万面元；下一阶段才考虑
CUDA 内核融合和受误差预算约束的分层快速远场。保留
`physical_accuracy_accepted=false`。
