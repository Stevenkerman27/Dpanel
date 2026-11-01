
import re, pathlib
p = pathlib.Path(r"D:\3D\Projects\panel\runs\fix_master\panel\p1_geometry_contact.py")
src = p.read_text(encoding="utf-8")
# 1. drop the certificate block in _classify_set
start = src.index("    if offending and wall_surface is not None:")
end = src.index("    grazing = 0", start)
src = src[:start] + src[end:]
# 2. drop the wall_surface parameter from the three signatures
src = src.replace("declared_region=None, graze_tol=0., wall_surface=None):",
                  "declared_region=None, graze_tol=0.):")
src = src.replace("declared_region=None, graze_tol=0., wall_surface=None)",
                  "declared_region=None, graze_tol=0.)")
# 3. drop _surface_distance
s = src.index("def _surface_distance(points, tris):")
e = src.index("def _classify_set(", s)
src = src[:s] + src[e:]
# 4. the certificate sentence in the docstring is gone with the param; nothing else to do
p.write_text(src, encoding="utf-8")
print("wall_surface left:", src.count("wall_surface"), "_surface_distance left:", src.count("_surface_distance"))
print("def count:", src.count("def _classify_set("), src.count("def classify_swept_wall("))
