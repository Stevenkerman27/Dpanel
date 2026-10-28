"""OpenVSP / VSPAERO client used by the DPanel verification work.

Provenance
----------
The geometry-side code (wing creation, tessellation plan, geometry-set
selection, compute-geometry and sweep drivers, polar reader) is a **port of
`DEP-OPT/infrastructure.py`** (the copy verified in production), trimmed to
what an external aerodynamic comparison needs.  Nothing else in DEP-OPT
(propellers, fuselage, control surfaces, FFD, mass, XFOIL coupling) is copied.

Two adaptations for the DPanel project:

1. **Panel vs VLM is an explicit mode.**  VSPAERO has no "analysis method"
   input; the mode follows the geometry sets (`GeomSet` / `ThinGeomSet`).
   `mode="panel"` puts everything in the thick set and nothing in the thin
   set, `mode="vlm"` does the opposite.  DEP-OPT's `SET_FIRST_USER` sets are
   *empty* in third-party models, which silently degrades panel mode to VLM.
2. **`LECluster` guard.**  Panel-mode `CLi` is wrong by orders of magnitude
   when the chordwise leading-edge clustering is too tight
   (see `docs/VSPAERO_PANEL_POLAR_DIAGNOSIS.md`).  `health` re-jects such
   cluster settings before meshing and flags implausible polar rows after.

Import layers
-------------
`vspapi.tessellate`, `vspapi.mesh`, `vspapi.results`, `vspapi.config`,
`vspapi.health` are pure Python/NumPy and import anywhere (the DPanel solver
environment has no OpenVSP).  Only `vspapi.api` touches `import openvsp`, and
it does so lazily.
"""

from .config import SweepConfig
from .health import check_polar_health, cluster_risk
from .mesh import VspGeom, measure, read_vspgeom
from .results import read_history, read_lod, read_polar
from .tessellate import make_wing_tess_plan, next_tess_value, wing_area

__all__ = [
    "SweepConfig",
    "VspGeom",
    "check_polar_health",
    "cluster_risk",
    "make_wing_tess_plan",
    "measure",
    "next_tess_value",
    "read_history",
    "read_lod",
    "read_polar",
    "read_vspgeom",
    "wing_area",
]
__version__ = "0.1.0"
