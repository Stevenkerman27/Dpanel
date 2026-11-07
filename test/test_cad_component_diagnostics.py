import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lifting3d"))
import cad_components


def test_unmatched_projection_diagnostic_is_json_serializable(monkeypatch):
    model = SimpleNamespace(getBoundary=lambda entities, **kwargs: [(2, 1)])
    gmsh = SimpleNamespace(model=model)
    monkeypatch.setattr(cad_components, "sample_surface_points",
                        lambda *args: np.zeros((3, 3)))
    monkeypatch.setattr(cad_components, "closest_trimmed_source_distance",
                        lambda *args: (float("inf"), None))
    tags, diagnostics = cad_components.classify_wall_surfaces(
        gmsh, [(3, 1)], {"wing": [(3, 2)]}, 1e-7)
    assert tags == {}
    assert diagnostics[0]["role_max_sample_residual"]["wing"] is None
    assert diagnostics[0]["unmatched_reason"] == "no_source_geometry_within_tolerance"
    json.dumps(diagnostics, allow_nan=False)
