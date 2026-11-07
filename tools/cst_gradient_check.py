"""Compare all sharp-TE GRASS CST derivatives with central differences."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from panel.p1_cst_gradient import fixed_wake_cst


parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--chord-panels", type=int, default=3)
parser.add_argument("--span-panels", type=int, default=3)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
code = torch.tensor([.17] * 8 + [-.17] * 8 + [0., 0.],
                    dtype=torch.float64, requires_grad=True)
options = dict(chord_panels=args.chord_panels, span_panels=args.span_panels)
loads, info = fixed_wake_cst(code, **options)
gradient = torch.stack([torch.autograd.grad(v, code, retain_graph=True)[0]
                        for v in loads]).detach().numpy()
rows = []
for step in (1e-4, 1e-5):
    difference = np.empty((2, 17))
    for i in range(17):
        delta = torch.zeros_like(code)
        delta[i] = step
        plus, plus_info = fixed_wake_cst(code.detach() + delta, **options)
        minus, minus_info = fixed_wake_cst(code.detach() - delta, **options)
        difference[:, i] = ((plus - minus) / (2 * step)).detach().numpy()
    error = np.abs(difference - gradient[:, :17])
    rows.append(dict(step=step, finite_difference=difference.tolist(),
                     max_absolute_error=error.max(axis=1).tolist(),
                     max_scaled_error=(error / np.maximum(np.abs(difference), 1e-6)).max(axis=1).tolist()))
report = dict(code=code.detach().tolist(), loads=loads.detach().tolist(),
              gradient=gradient.tolist(), forward=info, differences=rows,
              chord_panels=args.chord_panels, span_panels=args.span_panels,
              wake_panels=2, wake_length_chords=10, alpha_deg=4,
              physical_accuracy_accepted=False)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("CL/CD", report["loads"], "max errors",
      [row["max_absolute_error"] for row in rows], flush=True)
