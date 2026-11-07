"""Audit saved filament-wake snapshots for nonadjacent face intersections."""
import argparse
import json
from pathlib import Path

import numpy as np

from panel.p1_filament_free_wake import audit_nonadjacent_intersections


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out",type=Path,required=True)
parser.add_argument("snapshots",nargs="+",type=Path)
args=parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
report=dict(scope="Nonadjacent wake triangles only: transverse segment intersections and coplanar overlap; shared-node pairs and wall crossings not checked.",
            snapshots={})
for path in args.snapshots:
    with np.load(path) as data:
        pairs=audit_nonadjacent_intersections(data["xyz"],data["faces"])
    report["snapshots"][path.name]=dict(count=len(pairs),pairs=pairs)
    print("%s: %d nonadjacent intersections" % (path.name,len(pairs)),flush=True)
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
