import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Compact point indices in an existing P2 mesh without changing geometry.")
parser.add_argument("--mesh", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

mesh_path = args.mesh.resolve()
output_dir = args.out.resolve()
if output_dir.exists() and any(output_dir.iterdir()):
    parser.error(f"Refusing to overwrite non-empty output directory: {output_dir}")
metadata_path = mesh_path.with_name("mesh.json")
if not metadata_path.exists():
    parser.error(f"Adjacent metadata file is required: {metadata_path}")

source_mesh_hash = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
with np.load(mesh_path) as archive:
    arrays = {name: np.array(archive[name], copy=True) for name in archive.files}

points = arrays["points"]
tets = arrays["tets"]
triangles = arrays["boundary_triangles"]
tags = arrays["boundary_tags"]
boundary_tet = arrays["boundary_tet"] if "boundary_tet" in arrays else None
if points.ndim != 2 or points.shape[1] != 3:
    raise ValueError("points must have shape (n,3)")
if tets.ndim != 2 or tets.shape[1] != 10 or not np.issubdtype(tets.dtype, np.integer):
    raise ValueError("tets must be integer tet10 connectivity with shape (m,10)")
if triangles.ndim != 2 or triangles.shape[1] != 6 or not np.issubdtype(triangles.dtype, np.integer):
    raise ValueError("boundary_triangles must be integer tri6 connectivity with shape (b,6)")
if tags.shape != (len(triangles),):
    raise ValueError("boundary_tags must have one value per tri6")
if boundary_tet is not None and boundary_tet.shape != (len(triangles),):
    raise ValueError("boundary_tet must have one element index per tri6")
if np.any(tets < 0) or np.any(tets >= len(points)):
    raise ValueError("tet10 connectivity contains an out-of-range point index")
if np.any(triangles < 0) or np.any(triangles >= len(points)):
    raise ValueError("tri6 connectivity contains an out-of-range point index")

used_points = np.unique(tets)
if not np.all(np.isin(np.unique(triangles), used_points)):
    raise ValueError("a boundary triangle references a point unused by all tet10 elements")
point_count_before_compaction = len(points)
old_to_new = np.full(point_count_before_compaction, -1, dtype=np.int64)
old_to_new[used_points] = np.arange(len(used_points))
arrays["points"] = points[used_points]
arrays["tets"] = old_to_new[tets]
arrays["boundary_triangles"] = old_to_new[triangles]
removed_point_count = point_count_before_compaction - len(used_points)
if boundary_tet is not None and not np.array_equal(arrays["boundary_tet"], boundary_tet):
    raise RuntimeError("boundary_tet element adjacency changed during point compaction")

array_hash = hashlib.sha256()
for name in ["points", "tets", "boundary_triangles", "boundary_tags", "boundary_tet"]:
    if name in arrays:
        array_hash.update(np.ascontiguousarray(arrays[name]).tobytes())

output_dir.mkdir(parents=True, exist_ok=True)
output_mesh = output_dir / "mesh.npz"
np.savez_compressed(output_mesh, **arrays)
metadata["counts"]["points"] = len(arrays["points"])
metadata["counts"]["unused_points_removed"] = removed_point_count
metadata["counts"]["tets"] = len(arrays["tets"])
metadata["counts"]["boundary_triangles"] = len(arrays["boundary_triangles"])
metadata["counts"]["boundary_tets"] = len(arrays["boundary_tet"]) if boundary_tet is not None else None
metadata["sha256_arrays"] = array_hash.hexdigest()
metadata["mesh_compaction"] = {
    "method": "Exact point-index compaction using sorted unique tet10 point indices.",
    "source_mesh": str(mesh_path),
    "source_mesh_sha256": source_mesh_hash,
    "source_point_count": point_count_before_compaction,
    "compact_point_count": len(arrays["points"]),
    "removed_unused_point_count": removed_point_count,
    "coordinates_recomputed_or_moved": False,
    "tet10_indices_remapped": True,
    "tri6_indices_remapped": True,
    "boundary_tet_element_adjacency_preserved": boundary_tet is None or np.array_equal(arrays["boundary_tet"], boundary_tet),
    "source_boundary_tet_sha256": (
        hashlib.sha256(np.ascontiguousarray(boundary_tet).tobytes()).hexdigest()
        if boundary_tet is not None
        else None
    ),
}
metadata["mesh_sha256"] = hashlib.sha256(output_mesh.read_bytes()).hexdigest()
(output_dir / "mesh.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
print(
    json.dumps(
        {
            "output": str(output_dir),
            "source_mesh_sha256": source_mesh_hash,
            "output_mesh_sha256": metadata["mesh_sha256"],
            "source_points": point_count_before_compaction,
            "compact_points": len(arrays["points"]),
            "removed_unused_points": removed_point_count,
            "tet_count": len(arrays["tets"]),
            "boundary_tet_preserved": metadata["mesh_compaction"]["boundary_tet_element_adjacency_preserved"],
        },
        indent=2,
        ensure_ascii=True,
    )
)
