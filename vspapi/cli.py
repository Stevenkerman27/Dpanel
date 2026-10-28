"""Command line entry point: `python -m vspapi <command>`.

Commands
--------
`version`   print the OpenVSP API version and the resolved vspaero.exe
`inspect`   measure/validate a `.vspgeom`, optionally check a `.polar`
`sweep`     build a wing from a JSON spec, mesh it in the requested mode, solve it

The JSON spec for `sweep` (every key optional except the wing)::

    {"workdir": "runs/vsp_case", "mode": "panel", "name": "trap",
     "wing": {"spanlist": [1.0], "chordlist": [0.2, 0.1],
              "thick_camber": 0.12, "camber": 0.02, "camber_loc": 0.4,
              "te_cluster": 0.9, "le_cluster": 0.8},
     "mesh": {"tess_w": 53, "sect_tess_u": [121]},
     "sweep": {"alpha": [0, 2.5, 5, 7.5, 10], "vinf": 20, "rho": 1.225}}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_version(_args) -> int:
    from .api import find_vspaero_exe, openvsp_version
    from . import __version__

    api = openvsp_version()
    print("vspapi      :", __version__)
    print("OpenVSP API :", api)
    try:
        print("vspaero.exe :", find_vspaero_exe(api))
    except FileNotFoundError as exc:
        print("vspaero.exe :", "NOT FOUND -", exc)
    return 0


def _cmd_inspect(args) -> int:
    from .health import check_polar_health
    from .mesh import measure, mesh_quality, open_boundary_edges, read_vspgeom, validate_mesh
    from .results import read_polar

    geom = read_vspgeom(args.geom)
    info = measure(geom)
    info["quality"] = mesh_quality(geom)
    info["open_boundary_edges"] = int(open_boundary_edges(geom).shape[0])
    if args.strict:
        validate_mesh(geom)
    print(json.dumps(info, indent=2, default=float))
    if args.polar:
        rows = read_polar(args.polar)
        health = check_polar_health(rows)
        print(json.dumps(health, indent=2, default=float))
        return 0 if health["healthy"] else 2
    return 0


def _cmd_sweep(args) -> int:
    from .api import Case, WingSpec
    from .config import SweepConfig

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    case = Case(spec.get("workdir", args.workdir or "vsp_case"),
                name=spec.get("name", "trap"))
    wing = dict(spec.get("wing") or {})
    mesh = spec.get("mesh") or {}
    for key in ("tess_w", "sect_tess_u"):
        if key in mesh:
            wing[key] = mesh[key]
    wing_spec = WingSpec(**wing)
    planform = wing_spec.planform_area
    case.new_model()
    case.add_wing(wing_spec, tess_int=spec.get("tess_int", 0.0038),
                  allow_risky_cluster=args.allow_risky_cluster)
    case.write_model()
    mode = spec.get("mode", "panel")
    geom = case.compute_geometry(mode)
    info = case.assert_mode(geom, mode, planform)
    cfg_kwargs = dict(spec.get("sweep") or {})
    cfg = SweepConfig(sref=planform, cref=cfg_kwargs.pop("cref", wing_spec.chordlist[0]),
                      bref=cfg_kwargs.pop("bref", 2.0 * sum(wing_spec.spanlist)),
                      **cfg_kwargs)
    ncpu = int(spec.get("ncpu", 8))
    result = case.run(cfg, ncpu=ncpu)
    summary = {
        "mode": mode,
        "mesh": info,
        "planform_area": planform,
        "config": cfg.as_dict(),
        "exit_code": result["exit_code"],
        "seconds": result["seconds"],
        "ncpu": ncpu,
        "polar": result["polar"],
        "health": result["health"],
    }
    out = case.workdir / "vspapi_result.json"
    out.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("mode", "exit_code", "seconds")}, indent=2))
    print("mesh:", json.dumps(info, default=float))
    for row in result["polar"]:
        print("alpha=%-6.3g CLwtot=%-10.6g CDwtot=%-10.6g CLi=%-12.6g CDi=%-.6g"
              % (row.get("AoA", float("nan")), row.get("CLwtot", float("nan")),
                 row.get("CDwtot", float("nan")), row.get("CLi", float("nan")),
                 row.get("CDi", float("nan"))))
    if result["health"] and result["health"]["issues"]:
        print("WARNING:", "; ".join(result["health"]["issues"]), file=sys.stderr)
    print("wrote", out)
    return 0 if result["exit_code"] == 0 else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vspapi", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("version", help="print API / executable versions")
    p.set_defaults(func=_cmd_version)

    p = sub.add_parser("inspect", help="measure a .vspgeom (and check a .polar)")
    p.add_argument("geom")
    p.add_argument("--polar", default=None)
    p.add_argument("--strict", action="store_true")
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser("sweep", help="mesh + solve from a JSON spec")
    p.add_argument("spec")
    p.add_argument("--workdir", default=None)
    p.add_argument("--allow-risky-cluster", action="store_true")
    p.set_defaults(func=_cmd_sweep)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())