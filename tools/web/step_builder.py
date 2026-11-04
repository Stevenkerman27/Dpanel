"""Build wing.step and trailing_edge.json in a fresh interpreter.

gmsh.initialize() installs a SIGINT handler and therefore refuses to run in a
worker thread ("signal only works in main thread of the main interpreter"),
while the browser workbench builds inside an HTTP request thread.  So the CAD
step runs as its own process, which also keeps a gmsh crash away from the
server.  The child writes build.json; the parent only adds the display mesh.
"""
import argparse
import json
import pathlib
import sys

from tools.web import geometry
from tools.web import params as params_module


def main(argv=None):
    parser = argparse.ArgumentParser(description="DPanel wing STEP builder")
    parser.add_argument("--params", required=True, help="JSON file of raw parameters")
    parser.add_argument("--out", required=True, help="run directory")
    parser.add_argument("--count", type=int, default=41)
    args = parser.parse_args(argv)

    raw = json.loads(pathlib.Path(args.params).read_text(encoding="utf-8"))
    params, clamped = params_module.normalise(raw)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = geometry.write_step(out / "wing.step", params, args.count)
    geometry.write_trailing_edge(out / "trailing_edge.json", params)
    payload = dict(params=params, clamped=clamped,
                   reference=params_module.reference(params),
                   stations=params_module.full_stations(params),
                   trailing_edge=params_module.trailing_edge_points(params),
                   step=info,
                   files=dict(step=str(out / "wing.step"),
                              trailing_edge=str(out / "trailing_edge.json")))
    (out / "build.json").write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps(dict(volume=info["volume"], sections=info["sections"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
