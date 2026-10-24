# 封闭CAD输入协议

本入口读取OpenCASCADE支持的STEP/BREP/IGES实体，面向无环量闭体势流。它不推断机翼后缘，不附加Kutta或尾迹；导入飞机外形并不等于得到其升力解。

1. 只接受至少一个封闭实体，多个实体先作并集；相减后流体体积必须单一连通，未支持封闭内部流体腔。
2. 用原CAD联合包围盒中心和最大边长，将几何变换到单位参考长度。输出节点和求解坐标均为该归一化坐标，`native_origin/native_length`保留原坐标变换。`Sref`也必须使用归一化面积单位；原坐标和面积可按元数据恢复，不能混用。
3. 外部盒面tag100、所有物面tag10。CAD面拥有的tri6直接导出，不按三角形中心猜物面。完整保留tet10和tri6曲率；Gmsh节点排列显式转换，精确压缩未使用点。
4. 共享加载器检查所有P2边中点相容、165点Jacobian、边界完整性/邻接和流体连通性；这些为采样和拓扑检查，不是任意CAD网格质量的数学保证。

首次运行前规定工作流检查：平面STEP必须以“无封闭实体”明确拒绝；原生坐标平移且尺度相差1000倍的球体均须导入、产生有效网格并数值收敛。分别计算所有原始表面样本对无限域解析球体Cp的面积RMS，要求<0.05，原始壁面法向RMS<0.05，零压力力系数的CL/CD绝对值各<0.005。有限外盒是均匀Dirichlet而非解析边界，有限域误差包含在这里；这些门禁只认证本组球体工作流，不能认证任意CAD。

```powershell
python volume/build_cad_mesh.py --cad body.step --out runs/cad_body_mesh
python volume/solve_p2.py --mesh runs/cad_body_mesh/mesh.npz --mach 0 --sref 1 --out runs/cad_body_flow
```

OCC默认给出保守包围盒；周期B-spline的控制网范围可能大于几何范围。记录的
`native_bbox/native_bounds_method/native_origin_definition`明确这个约定；包围盒中心
不是质量中心或解析几何中心。归一化只用于可逆坐标/尺度变换，不能将控制网尺寸
当作飞机参考弦长。椭球参考使用记录的映射中心，旧中心/长度门禁的失败原样保留。

导入拥有volume标签并不保证周期面边界可网格化。空物面会在P2构网之前明确拒绝，
记录面编号和三角形数；不能在没有物面单元时继续使用空流体解。单面椭球STEP
失败与分体重放见 `ELLIPSOID_RESULTS.md` 和 `audit/cad_import_*`。
