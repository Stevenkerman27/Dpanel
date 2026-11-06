"""Write the GRASS fixed-topology mesh in an isolated worker process."""
import argparse
import json
from pathlib import Path

from panel.grass_mesh import build_grass_mesh

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--size", type=float, required=True)
args = parser.parse_args()
params = json.loads(args.input.read_text(encoding="utf-8"))
data = build_grass_mesh(params["grass_sections"], args.size)
args.out.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
print("GRASS fixed mesh: %d wall faces, chord=%d, span/segment=%d" %
      (len(data["faces"]), data["chord_panels"], data["span_panels_per_segment"]))
