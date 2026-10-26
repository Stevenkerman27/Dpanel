# 可选几何特征局部加密（研究PR，非物理精度通过）

Issue #42。此更改仅增加 Gmsh 2D 原生网格尺寸**回调**，无新依赖、无湿壁/尾迹实体变更、无解器更改。默认不传 `--mesh-features` 时，旧 `cad_half.py` 路径不变。

## 输入示例（SI 原始 CAD 坐标，不是归一化坐标）

```json
{
  "schema": "dpanel.mesh_features.v1",
  "units": "m",
  "features": [
    {
      "name": "leading_edge",
      "points": [[0.0, 0.0, 0.0], [0.1, 1.0, 0.0], [0.2, 2.0, 0.0]],
      "target_ratio": 0.25,
      "radius_chords": 0.12
    },
    {
      "name": "trailing_edge",
      "points": [[1.0, 0.0, 0.0], [1.1, 1.0, 0.0], [1.2, 2.0, 0.0]],
      "target_ratio": 0.5,
      "radius_chords": 0.08
    }
  ]
}
```

以上只是**示例，不表示任何已有 CAD 真实几何**。LE/TE 特征线须由原始 GRASS/VSP/CAD 参数显式生成并检查与 STEP 真实曲线相符；切勿按全局最小 x 猜后掠前缘。对 BWB、翼身交界或多翼组，输入可以包含多个具有唯一名字的特征折线。

`target_ratio` 在 [1/8,1]，相对于本次 `--size`（参考弦归一化）给定目标局部尺寸；`radius_chords` 是相对于 **TE 声明 reference_chord** 的特征带物理宽度。函数采用距三维折线的最近距离、C1 smoothstep 过渡，不把更大 lc 强加给原背景场。多个特征取最小尺寸。请求的尺寸**不等于**实际生成的面元尺寸/质量，必须测量。

## 使用

```powershell
python -m panel.cad_half --wing-cad <ORIGINAL.step> --trailing-edge <ORIGINAL_TE.json> --mesh-features <ACTUAL_LE_TE_FEATURES.json> --out runs/NEW_FEATURE_MESH --size 0.10 --far-size 1.5 --radius 5 --alpha-deg 5
```

`cad_half.py` 将坐标按原参考弦、原点归一化后交给 Gmsh `model.mesh.setSizeCallback(dim,tag,x,y,z,lc)`。既有湿壁 Distance+Threshold 背景场不改；size回调对曲线和面都起作用，避免只有面内加密但几何棱边仍粗。源文件 SHA-256 和每条曲线设置写入新 surface provenance。所有原生 OCC fragment、共形接触、物理 wall/wake 选面及镜像拓扑约束仍需原验证器检查。

## 本地先行门槛与独立A/B

1. 纯数学单测；Gmsh 版本、原 CAD/TE/feature SHA，新增 CLI 触发旧 v1 一致性回归。无 flag 下需与相同环境旧算例数值完全一致（不要把网格编号跨环境字节差当物理退化）。
2. 在一个**同一原始 STEP** 上生成：uniform size .10、LE-only (ratio .25, band .12c)、LE+TE (ratio .25/.5)、全局 .075/.05。不同组必须各独立新目录，记录实际LE/TE带内单元/面积/first edge length、三角角度/质量/法向跨度和 true TE-wake共边，且必须未穿墙、无对称面 cap。
3. 比较 CL/Cp截面/CM/CD压力 *原值*、真实无穿透/尾迹压力跳跃/原始Cp极值坐标、资源RSS。不能只看 CD 变大变小；不拿 CL 相邻 <5% 当网格无关。
4. 固定代表例，再做 `wall×Gamma×wake length` 独立扫描。带相消和尖峰时，不做未经渐近区确认的 Richardson/GCI 外推。超预算应明确拒绝。
5. 多机翼 `shed_groups` 属另一个PR/issue (#41)。**不能同一次实验同时更改涡片物理模型和网格策略**。

## 范围与诚实限制

本云端环境有 NumPy/pytest，**没有 Gmsh**。已完成纯数学策略检查，但不能在这里声称新尺寸回调真实 CAD 网格通过。实际生成的网格元素密度、不同 OpenCascade 曲线参数化、稳定性与资源预算只能由本地 Gmsh 回归验收。此PR不实现 CAD 曲率驱动或后验自适应 `PostView`；等第一轮实际结果后，在独立PR接入。保留 `physical_accuracy_accepted=false`；不合并、不开启训练。
