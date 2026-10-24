# 原始压力力与力矩

`loads.py`读取已收敛的 `field.npz`，用已保存的物理坐标、原始Cp、流体域
外法向和面积权重逐高斯点积分。总量与组件量使用相同参考面积和根弦：

```
CF = sum(Cp * n_fluid * dS) / Sref
CM = sum((x-center) cross (Cp * n_fluid) * dS) / (Sref * root_chord)
```

物面流体外法向指向实体内部，因此上述力方向就是流体对实体的压力力。
系数分量采用全局右手x/y/z坐标，不附加飞机俯仰等符号重定义。
默认力矩中心是物理坐标中的翼根四分之一弦点；`--center X Y Z`使用保存网格
的坐标单位。统一入口的 `numerics.moment-center` 使用同一单位。中心、根弦、
Sref、坐标约定与原始文件哈希均写入输出。

```powershell
python lifting3d/loads.py --run RUN --mesh MESH --center 0.25 0 0 --out loads.json
python -m unittest lifting3d.test_loads -v
```

解析检查使用平移长方体上的线性压力。由散度定理独立得到整面积分力与力矩，
再检查参考点平移恒等式、1000倍尺度变换和组件力矩闭合。实际厚翼身原始场
回算压力力与求解结果完全一致，组件力/力矩求和误差分别约6.0e-15/5.1e-16。
这核验了后处理公式、符号和量纲；不会将未验收的压力场转变成可信设计载荷。
