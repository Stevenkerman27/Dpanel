
import io, pathlib
p = pathlib.Path(r"D:\3D\Projects\panel\runs\fix_master\panel\p1_geometry_contact.py")
s = p.read_text(encoding="utf-8")
a = s.index("def _hull_triangles(points):")
b = s.index("def _surface_distance(points, tris):")
s2 = s[:a]+s[b:]
p.write_text(s2, encoding="utf-8", newline="\n")
print("removed", b-a, "chars; py compile:", end=" ")
import py_compile; py_compile.compile(str(p), doraise=True); print("ok")
