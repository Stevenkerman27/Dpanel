"""OpenVSP API layer: the only module that imports `openvsp`.

Ported from `DEP-OPT/infrastructure.py` (wing creation, geometry-set
selection, compute-geometry, sweep).  Two deliberate differences:

* **Mode is explicit.**  VSPAERO 3.x has no "analysis method" input; the mode
  follows the geometry sets.  DEP-OPT relies on `SET_FIRST_USER` sets that
  only exist for models *it* created; feeding a third-party `.vsp3` that way
  silently falls back to VLM.  Here `mode="panel"` puts the wing in the
  thick set (`GeomSet=SET_FIRST_USER, ThinGeomSet=SET_NONE`) and `mode="vlm"`
  the other way round, with `assert_mode()` checking the wetted/planform area
  ratio of the generated mesh afterwards.
* **Cluster guard.**  `add_wing` refuses `LECluster < 0.5` unless explicitly
  allowed, because that setting makes panel-mode `CLi` wrong by ~100x.

Requires an OpenVSP Python install (this project uses the `vsppytools` conda
env, OpenVSP >= 3.52).  The solver environment `.conda_env` has no OpenVSP and
must not import this module.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import SweepConfig
from .health import check_polar_health, cluster_risk
from .mesh import VspGeom, measure, read_vspgeom, validate_mesh
from .results import read_history, read_lod, read_polar
from .tessellate import make_wing_tess_plan, next_tess_value, wing_area

REPO_ROOT = Path(__file__).resolve().parents[1]

_VSP = None


def _candidate_install_roots():
    """OpenVSP install directories to fall back on, newest first."""
    roots = []
    env = os.environ.get("VSP_ROOT")
    if env:
        roots.append(Path(env))
    for base in (REPO_ROOT, REPO_ROOT.parent):
        if base.is_dir():
            roots.extend(base.glob("OpenVSP-*-win64"))

    def version_key(path):
        match = re.search(r"OpenVSP-(\d+)\.(\d+)\.(\d+)", path.name)
        return tuple(int(v) for v in match.groups()) if match else (0, 0, 0)

    unique = {p.resolve() for p in roots if (p / "python").is_dir()}
    return sorted(unique, key=version_key, reverse=True)


def _import_from_install(root: Path):
    """Import `openvsp` out of an unpacked OpenVSP tree (<root>/python/...)."""
    import importlib

    sys.path.insert(0, str(root / "python" / "openvsp"))
    for extra in ("degen_geom", "utilities", "openvsp_config"):
        candidate = root / "python" / extra
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(root))
        except OSError:  # pragma: no cover - non-Windows
            pass
    module = importlib.import_module("openvsp")
    if not hasattr(module, "GetVSPVersion"):
        raise ImportError(f"{root} did not provide a usable openvsp module")
    return module


def load_openvsp():
    """Import `openvsp`, preferring the installed package, falling back to a tree.

    A conda install whose `site-packages/openvsp` directory is unreadable (ACL
    damage, or a partially removed upgrade) imports as an *empty namespace
    package*: `import openvsp` succeeds and every attribute is missing.  That
    case is detected and the unpacked `OpenVSP-*-win64` tree next to the
    repository is used instead.  Set `VSP_ROOT` to force a specific install.
    """
    global _VSP
    if _VSP is not None:
        return _VSP
    first_error = None
    try:
        import openvsp as vsp  # type: ignore

        if hasattr(vsp, "GetVSPVersion"):
            _VSP = vsp
            return _VSP
        first_error = ImportError(
            "imported 'openvsp' has no GetVSPVersion: the site-packages copy is "
            "an empty namespace package (unreadable directory?)")
    except ImportError as exc:  # pragma: no cover - environment dependent
        first_error = exc
    sys.modules.pop("openvsp", None)
    errors = [str(first_error)]
    for candidate in _candidate_install_roots():
        try:
            _VSP = _import_from_install(candidate)
            return _VSP
        except Exception as exc:  # pragma: no cover - environment dependent
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError(
        "openvsp is not usable. Use the OpenVSP conda env (vsppytools), or point "
        "VSP_ROOT at an unpacked OpenVSP-*-win64 install. Details: " + " | ".join(errors))


def openvsp_version() -> str:
    return load_openvsp().GetVSPVersion()


def find_vspaero_exe(version: Optional[str] = None, *, extra_roots: Sequence = ()) -> Path:
    """Locate `vspaero.exe`, preferring an install matching the API version."""
    env = os.environ.get("VSPAERO_EXE")
    if env and Path(env).is_file():
        return Path(env)
    roots = [REPO_ROOT, REPO_ROOT.parent, *[Path(p) for p in extra_roots]]
    candidates = []
    for base in roots:
        if not base.is_dir():
            continue
        for path in base.glob("OpenVSP-*-win64/vspaero.exe"):
            match = re.search(r"OpenVSP-(\d+)\.(\d+)\.(\d+)", str(path))
            key = tuple(int(v) for v in match.groups()) if match else (0, 0, 0)
            candidates.append((key, path))
    if not candidates:
        raise FileNotFoundError(
            "vspaero.exe not found; set VSPAERO_EXE or keep an OpenVSP-*-win64 tree "
            "next to the repository")
    candidates.sort(key=lambda item: item[0])
    if version:
        want = tuple(int(v) for v in re.findall(r"\d+", version)[:3])
        for key, path in candidates:
            if key == want:
                return path
    return candidates[-1][1]


# The wing is registered in SET_FIRST_USER by `Case.add_wing` (exactly like
# DEP-OPT's create_wing).  Panel mode = "thick set only", VLM = "thin set only";
# SET_ALL/SET_NONE do *not* express this, a model without set membership solves
# as VLM no matter which of the two you put in GeomSet.
MODE_SETS = {"panel": ("SET_FIRST_USER", "SET_NONE"),
             "vlm": ("SET_NONE", "SET_FIRST_USER")}

# Wetted area / full planform area: thick = both surfaces, thin = one sheet.
WETTED_PER_PLANFORM = {"panel": 1.0, "vlm": 0.5}


def geometry_sets(mode: str) -> tuple:
    """(GeomSet, ThinGeomSet) for `mode`, resolved against the loaded module."""
    if mode not in MODE_SETS:
        raise ValueError(f"mode must be one of {sorted(MODE_SETS)}, got {mode!r}")
    vsp = load_openvsp()
    thick_name, thin_name = MODE_SETS[mode]
    return int(getattr(vsp, thick_name)), int(getattr(vsp, thin_name))


def _omp_threads(stdout: str) -> Optional[int]:
    """Read the thread count back from VSPAERO's banner ("NumberOfThreads_: N")."""
    for line in stdout.splitlines():
        if "NumberOfThreads_:" in line:
            try:
                return int(line.split(":", 1)[1].strip())
            except (IndexError, ValueError):
                return None
    return None


@dataclass
class WingSpec:
    spanlist: Sequence[float]
    chordlist: Sequence[float]
    twistlist: Sequence[float] = ()
    sweeploc: float = 0.0
    thick_camber: float = 0.12
    camber: float = 0.02
    camber_loc: float = 0.4
    airfoil_file: str = ""
    name: str = "wing"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    y_rot: float = 0.0
    te_cluster: float = 0.9
    le_cluster: float = 0.8
    max_growth: Optional[float] = None
    tess_w: Optional[int] = None
    sect_tess_u: Optional[Sequence[int]] = None

    @property
    def planform_area(self) -> float:
        return wing_area(self.spanlist, self.chordlist)

    def tessellation(self, tess_int: float) -> dict:
        plan = make_wing_tess_plan(self.spanlist, self.chordlist, tess_int)
        if self.tess_w is not None:
            plan["Tess_W"] = int(self.tess_w)
        if self.sect_tess_u is not None:
            plan["SectTess_U"] = [int(v) for v in self.sect_tess_u]
        if len(plan["SectTess_U"]) != len(self.spanlist):
            raise ValueError("SectTess_U must have one entry per span section")
        return plan


class Case:
    """One VSPAERO case: a working directory, a `.vsp3` model and the exe."""

    def __init__(self, workdir, name: str = "trap", exe: Optional[Path] = None):
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.exe = Path(exe) if exe else find_vspaero_exe(openvsp_version())
        self._wing_count = 0

    # ----------------------------------------------------------------- model
    @property
    def vsp(self):
        return load_openvsp()

    def new_model(self) -> None:
        vsp = self.vsp
        vsp.ClearVSPModel()
        vsp.SetSetName(int(getattr(vsp, "SET_FIRST_USER", 3)), "wing")
        self._wing_count = 0

    def add_wing(self, spec: WingSpec, *, tess_int: float = 0.0038,
                 allow_risky_cluster: bool = False) -> str:
        risk = cluster_risk(spec.te_cluster, spec.le_cluster, spec.max_growth)
        if risk["risk"] == "high" and not allow_risky_cluster:
            raise ValueError(
                "Refusing LECluster=%.3g: %s. Pass allow_risky_cluster=True to override "
                "(panel-mode CLi will be wrong)." % (spec.le_cluster, risk["reason"]))
        vsp = self.vsp
        plan = spec.tessellation(tess_int)
        wid = vsp.AddGeom("WING", "")
        vsp.SetGeomName(wid, spec.name)
        vsp.SetParmVal(wid, "X_Rel_Location", "XForm", float(spec.x))
        vsp.SetParmVal(wid, "Y_Rel_Location", "XForm", float(spec.y))
        vsp.SetParmValUpdate(wid, "Z_Rel_Location", "XForm", float(spec.z))
        vsp.SetParmValUpdate(wid, "Y_Rel_Rotation", "XForm", float(spec.y_rot))
        vsp.SetParmVal(vsp.FindParm(wid, "Sym_Planar_Flag", "Sym"), 0)
        vsp.SetParmVal(wid, "Tess_W", "Shape", plan["Tess_W"])
        vsp.SetParmValUpdate(wid, "TECluster", "WingGeom", float(spec.te_cluster))
        vsp.SetParmValUpdate(wid, "LECluster", "WingGeom", float(spec.le_cluster))
        if spec.max_growth is not None:
            vsp.SetParmValUpdate(wid, "MaxGrowth", "WingGeom", float(spec.max_growth))
        n_sections = len(spec.spanlist)
        for _ in range(n_sections):
            vsp.InsertXSec(wid, 1, vsp.XS_FOUR_SERIES)
        vsp.CutXSec(wid, 1)
        vsp.SetParmVal(wid, "Root_Chord", "XSec_1", float(spec.chordlist[0]))
        self._apply_airfoil(wid, 0, spec)
        vsp.Update()
        twistlist = list(spec.twistlist) or [0.0] * n_sections
        for i in range(1, n_sections + 1):
            sec_index = "XSec_%d" % i
            tess_u = plan["SectTess_U"][i - 1]
            vsp.SetParmVal(wid, "SectTess_U", sec_index, tess_u)
            vsp.SetParmVal(wid, "InCluster", sec_index,
                           float(spec.chordlist[i - 1] / spec.chordlist[i]))
            vsp.SetDriverGroup(wid, i, vsp.SPAN_WSECT_DRIVER,
                               vsp.ROOTC_WSECT_DRIVER, vsp.TIPC_WSECT_DRIVER)
            vsp.SetParmVal(wid, "Span", sec_index, float(spec.spanlist[i - 1]))
            vsp.SetParmVal(wid, "Tip_Chord", sec_index, float(spec.chordlist[i]))
            vsp.SetParmVal(wid, "Twist", sec_index, float(twistlist[i - 1]))
            vsp.SetParmVal(wid, "Sweep_Location", sec_index, float(spec.sweeploc))
            self._apply_airfoil(wid, i, spec)
            vsp.Update()
        vsp.SetSetFlag(wid, int(getattr(vsp, "SET_FIRST_USER", 3)), True)
        self._wing_count += 1
        return wid

    def _apply_airfoil(self, wid: str, index: int, spec: WingSpec) -> None:
        vsp = self.vsp
        if spec.airfoil_file:
            surf = vsp.GetXSecSurf(wid, index)
            vsp.ChangeXSecShape(surf, index, vsp.XS_FILE_AIRFOIL)
            vsp.ReadFileAirfoil(vsp.GetXSec(surf, index), spec.airfoil_file)
            return
        curve = "XSecCurve_%d" % index
        vsp.SetParmVal(wid, "ThickChord", curve, float(spec.thick_camber))
        vsp.SetParmVal(wid, "Camber", curve, float(spec.camber))
        vsp.SetParmValUpdate(wid, "CamberLoc", curve, float(spec.camber_loc))

    def write_model(self) -> Path:
        vsp = self.vsp
        vsp.Update()
        path = self.workdir / (self.name + ".vsp3")
        vsp.WriteVSPFile(str(path), int(getattr(vsp, "SET_ALL", 0)))
        return path

    # -------------------------------------------------------------- geometry
    def compute_geometry(self, mode: str = "panel", *,
                         symmetry: int = 2) -> VspGeom:
        vsp = self.vsp
        thick, thin = geometry_sets(mode)
        vsp.WriteVSPFile(str(self.workdir / (self.name + ".vsp3")),
                         int(getattr(vsp, "SET_ALL", 0)))
        analysis = "VSPAEROComputeGeometry"
        vsp.SetAnalysisInputDefaults(analysis)
        vsp.SetIntAnalysisInput(analysis, "Symmetry", [int(symmetry)], 0)
        vsp.SetIntAnalysisInput(analysis, "GeomSet", [thick], 0)
        vsp.SetIntAnalysisInput(analysis, "ThinGeomSet", [thin], 0)
        result = vsp.ExecAnalysis(analysis)
        try:
            path = vsp.GetStringResults(result, "VSPGeomFileName")[0]
        except Exception:  # pragma: no cover - very old builds
            path = vsp.GetStringResults(result, "DegenGeomFileName")[0]
        geom = read_vspgeom(path)
        validate_mesh(geom)
        return geom

    def assert_mode(self, geom: VspGeom, mode: str, planform_area: float,
                    *, rtol: float = 0.2) -> dict:
        """Check the generated mesh actually is the requested discretisation.

        `planform_area` is the *full-aircraft* planform (semi-span wing area
        doubled, i.e. `wing_area(spanlist, chordlist)` for a half-span model).
        A thick (panel) mesh covers both surfaces -> wetted/planform ~ 1; a thin
        (VLM) sheet covers one camber surface -> ~ 0.5.
        """
        info = measure(geom)
        ratio = info["area"] / float(planform_area)
        expected = WETTED_PER_PLANFORM[mode]
        if abs(ratio - expected) > rtol * expected:
            raise RuntimeError(
                "Generated mesh does not match mode %r: area/planform = %.3f "
                "(expected ~%.2f -> %s). In VSPAERO the mode follows the geometry "
                "sets, and a wing that is in neither set falls back to VLM."
                % (mode, ratio, expected,
                   "thick panel mesh" if expected > 0.75 else "thin VLM sheet"))
        info.update({"mode": mode, "area_over_planform": ratio})
        return info

    # ---------------------------------------------------------------- solve
    def run(self, cfg: SweepConfig, *, timeout: Optional[float] = None,
            extra_args: Sequence[str] = (), ncpu: Optional[int] = 8) -> dict:
        """Run `vspaero` in `workdir`.

        Threads come from VSPAERO's own command-line option
        (`vspaero [options] <FileName>`, `-omp <N>  Use 'N' processes.`), which
        calls `omp_set_num_threads`.  The `.vspaero` case file has **no** NCPU
        entry and the environment variables are ignored by the solver itself, so
        passing `-omp` is the only channel that works (measured: 66.1 s at 1
        thread, 14.2 s at 8, 11.0 s at 32 for a 6344-panel thick case, identical
        CL).  The OMP/OPENBLAS/MKL environment variables are still capped so that
        several cases can run side by side without BLAS allocation failures.
        """
        cfg.write(self.workdir / (self.name + ".vspaero"))
        env = dict(os.environ)
        if ncpu:
            # VSPAERO parallelises through OpenMP; the BLAS-backed solves inside
            # it honour OPENBLAS_NUM_THREADS.  Both must be capped when several
            # cases run side by side, otherwise OpenBLAS dies with
            # "Memory allocation still failed after 10 retries".
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                env[key] = str(int(ncpu))
        argv = [str(self.exe)]
        if ncpu:
            argv += ["-omp", str(int(ncpu))]
        argv += [self.name, *extra_args]
        started = time.time()
        proc = subprocess.run(
            argv, cwd=str(self.workdir),
            capture_output=True, text=True, timeout=timeout, env=env)
        elapsed = time.time() - started
        (self.workdir / "stdout.log").write_text(proc.stdout, encoding="utf-8",
                                                 errors="ignore")
        polar_path = self.workdir / (self.name + ".polar")
        rows = read_polar(polar_path) if polar_path.exists() else []
        history_path = self.workdir / (self.name + ".history")
        lod_path = self.workdir / (self.name + ".lod")
        return {
            "exit_code": proc.returncode,
            "seconds": elapsed,
            "ncpu": ncpu,
            "omp_threads": _omp_threads(proc.stdout),
            "polar": rows,
            "history": read_history(history_path) if history_path.exists() else [],
            "lod": read_lod(lod_path, len(cfg.alpha)) if lod_path.exists() else [],
            "health": check_polar_health(rows) if rows else None,
            "stdout_tail": proc.stdout.strip().splitlines()[-8:],
            "workdir": str(self.workdir),
        }