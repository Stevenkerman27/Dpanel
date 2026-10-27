# vspapi — OpenVSP / VSPAERO client (project copy)

One small, tested client for the external VSPAERO comparisons. It replaces the
ad-hoc scripts under `runs/vsp_conv/` and the verbatim DEP-OPT copy in
`runs/vsp_conv/depopt/`.

## Why a copy exists at all

`DEP-OPT/infrastructure.py` (93 kB) is the *verified* implementation, but it is
inseparable from DEP-OPT's aircraft-design workflow: propellers, fuselages,
control surfaces, FFD, mass estimates, XFOIL coupling, a global mutable
`case_name`, and `import prop / stall_limit / spanwise_lift` at module level.
Importing it from here drags in SciPy + Matplotlib + pandas and a working-
directory contract that a solver test cannot honour.

`vspapi` keeps **only** the VSP-facing 20%: wing tessellation, model building,
geometry-set / mode selection, compute-geometry, sweep, and the four result
parsers. ~800 lines, no global state, importable without OpenVSP unless you
touch `vspapi.api`.

## Layers

| module | needs OpenVSP? | contents |
| --- | --- | --- |
| `vspapi.tessellate` | no | `next_tess_value`, `make_wing_tess_plan`, `wing_area` (DEP-OPT port) |
| `vspapi.config` | no | `SweepConfig` <-> `.vspaero` text, byte-stable writer |
| `vspapi.mesh` | no | `.vspgeom` reader/writer, area/quality/boundary audit, y=0 mirror with welding |
| `vspapi.results` | no | `.polar`, `.history`, `.lod` (block-per-alpha), `.tec.dat` parsers |
| `vspapi.health` | no | `LECluster` gate, `CLi`/`CLwtot` consistency gate |
| `vspapi.api` | **yes** | `Case`: `new_model`, `add_wing`, `compute_geometry(mode)`, `run(cfg)`, `assert_mode` |
| `vspapi.cli` | for `sweep` | `python -m vspapi version / inspect / sweep` |

Pure layers are covered by `test/test_vspapi_core.py` (17 tests, no OpenVSP).

## The two adaptations

### 1. Panel vs VLM is explicit

VSPAERO has no analysis-method input; the mode follows the **geometry sets**:

| mode | `GeomSet` | `ThinGeomSet` | mesh | wetted/planform | `Model type` |
| --- | --- | --- | --- | --- | --- |
| `panel` | `SET_FIRST_USER` (3) | `SET_NONE` (-1) | thick, both surfaces | ~1.02 | `PANEL or MIXED` |
| `vlm` | `SET_NONE` (-1) | `SET_FIRST_USER` (3) | one camber sheet | ~0.50 | `VLM` |

Two traps, both of which cost real time here:

* A wing that belongs to **neither** set is meshed as VLM no matter what you
  pass in `GeomSet` — `SET_ALL` (0) does *not* mean "panel". `Case.add_wing`
  therefore calls `SetSetFlag(wid, SET_FIRST_USER, True)` exactly like DEP-OPT's
  `create_wing`, and `Case.assert_mode()` re-checks the generated mesh
  (wetted area / full planform ~ 1.02 for panel, ~ 0.5 for VLM).
* A wing in **both** sets is meshed twice and VSPAERO warns
  `In both the thick and the thin geometry set, so meshed twice`.

### 2. Clustering guard

In panel mode `CLi` (trailing-edge Kutta-Joukowski) is wrong by up to 200x when
the chordwise leading-edge clustering is tight; see
`docs/VSPAERO_PANEL_POLAR_DIAGNOSIS.md` section 4.3. Measured with everything
else identical: `LECluster 0.25 -> CLi 125.45`, `0.50 -> 5.50`,
`0.80 -> 0.635` (`CLwtot` stays 0.65-0.67 throughout). `add_wing` refuses
`LECluster < 0.5` unless `allow_risky_cluster=True`, and
`check_polar_health` flags rows where `|CLi / CLwtot| > 1.5`.

## Environment notes

* **Only `vspapi.api` needs OpenVSP.** The solver env `.conda_env` has none,
  which is deliberate; the pure layers and their tests run there.
* A broken conda install (empty namespace package, ACL-denied
  `site-packages/openvsp` — the case for `vsppytools` 3.55 here) is detected
  and `load_openvsp()` falls back to an unpacked `OpenVSP-*-win64` tree, newest
  version first, or to `VSP_ROOT` when that variable is set.
* `.vspgeom` v3 in OpenVSP 3.55 inserts an **extra panel-count line** between the
  points and the panel records (DEP-OPT's reader always assumed it; 3.52 files
  may or may not carry it). The parser accepts both layouts.

## Usage

```powershell
$env:VSP_ROOT = 'D:/3D/Projects/OpenVSP-3.55.0-win64'
& C:/Users/zyx20/anaconda3/envs/vsppytools/python.exe -m vspapi version
& C:/Users/zyx20/anaconda3/envs/vsppytools/python.exe -m vspapi inspect runs/vsp_conv/pth_52x120/trap.vspgeom --polar runs/vsp_conv/pth_52x120/trap.polar
```

```python
from vspapi.api import Case, WingSpec
from vspapi.config import SweepConfig

case = Case("runs/vsp_case", name="trap")
case.new_model()
spec = WingSpec(spanlist=[1.0], chordlist=[0.2, 0.1],
                tess_w=53, sect_tess_u=[121])       # NACA 2412 by default
case.add_wing(spec)
geom = case.compute_geometry("panel")
print(case.assert_mode(geom, "panel", spec.planform_area))
cfg = SweepConfig(sref=spec.planform_area, alpha=[0, 2.5, 5, 7.5, 10],
                  vinf=20.0, rho=1.225)
result = case.run(cfg)                             # polar / lod / history / health
for row in result["polar"]:
    print(row["AoA"], row["CLwtot"], row["CLi"], result["health"]["healthy"])
```

## Migration note

The long-term home for OpenVSP glue is the shared `aircraft-tools` package
(`D:/3D/Projects/aircraft-tools`, currently a 52 kB copy of GRASS-plane's
`openvsp_infrastructure.py`). `vspapi` is deliberately dependency-free and
state-free so it can be lifted there verbatim (rename the package, keep the
module split) and DEP-OPT/GRASS can then import it instead of carrying their own
copies. Nothing here imports DEP-OPT at runtime; the relationship is provenance
only.
