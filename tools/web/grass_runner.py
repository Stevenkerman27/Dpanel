"""Fixed-topology GRASS-section P1 solve and design gradients."""
import argparse
import json
from pathlib import Path

import torch

from panel.grass_mesh import reconstruct_grass_mesh
from panel.p1_cst_gradient import fixed_wake_geometry


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--alpha", type=float, default=4.)
parser.add_argument("--kutta", default="production",
                    choices=["production", "pressure", "potential_jump"])
args = parser.parse_args()
data = json.loads(args.input.read_text(encoding="utf-8"))
names = ("leading_edge_xyz", "chord", "twist_radians", "cst")
geometry, wing, xyz, wake, area = reconstruct_grass_mesh(data)
kutta = None if args.kutta == "production" else args.kutta
values, info = fixed_wake_geometry(wing, xyz, wake, area, args.alpha, 160, kutta=kutta)
seeds = torch.eye(values.numel(), dtype=values.dtype, device=values.device)
derivatives = torch.autograd.grad(values, tuple(geometry.values()), seeds, is_grads_batched=True)
gradients = {}
for row, label in enumerate(("CL", "CD")):
    gradients[label] = {name: derivative[row].detach().tolist()
                        for name, derivative in zip(names, derivatives)}
    gradients[label]["leading_edge_xyz"][0] = [None, None, None]
    for section in gradients[label]["cst"]:
        section[-1] = None
result = dict(mode="fixed", wall_formulation="galerkin", CL=float(values[0].detach()),
              CD=float(values[1].detach()), panels=len(wing.faces), gradients=gradients,
              gradient_constraints="根前缘坐标和各截面尾缘厚度固定，不提供其梯度。",
              warning="使用当前正式固定拓扑网格；气动网格收敛和物理精度尚未验收。",
              **info)
args.out.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
