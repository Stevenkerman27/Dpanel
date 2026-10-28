"""Sample the spanwise circulation shape Gamma(y) for each nested Gamma level.

Read-only: the module panel.gamma_refinement is used unmodified; this script only
re-assembles the augmented system and evaluates the refined basis on one common
station grid, so the three levels can be compared point by point.
"""
import argparse, json, pathlib
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from panel.run_half_native import load_half_surface
from panel.half_native import solve_half_fields
from panel.gamma_refinement import (refine_space, basis_at, split_wake, weak_te,
                                    wake_operator, solve_augmented)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surface", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--subdivisions", type=int, default=32)
    a = p.parse_args()
    fixture = json.loads(a.surface.read_text(encoding="utf-8"))
    inputs = load_half_surface(a.surface)
    ref = jax.jit(solve_half_fields)(**inputs)
    ref["CL"].block_until_ready()
    A = np.asarray(ref["matrix_half"])
    rhs0 = np.asarray(ref["rhs_half"])
    nf = len(fixture["faces"])
    body, rhs = A[:nf, :nf].copy(), rhs0[:nf].copy()
    x = np.asarray(inputs["vertices"])
    t = inputs["topology"]
    centers = x[np.asarray(t.faces)].mean(axis=1)
    ends = x[np.asarray(t.te_edges)]
    up, lo = np.asarray(t.upper_te), np.asarray(t.lower_te)
    y0 = np.asarray(fixture["gamma_stations"])
    dofs0 = np.asarray(fixture["gamma_station_dof"], dtype=np.int64)
    grid = np.concatenate([np.linspace(l, r, a.subdivisions + 1)[:-1]
                           for l, r in zip(y0[:-1], y0[1:])] + [y0[-1:]])
    out = dict(surface=str(a.surface), wall_panels=nf, stations_y=y0.tolist(),
               sample_y=grid.tolist(), levels=[])
    for factor in (1, 2, 4):
        space = refine_space(y0, dofs0, factor)
        pieces, parents, interval, geom = split_wake(
            x[np.asarray(t.wake_faces)], space.stations, max_pieces=20000)
        te = weak_te(ends, up, lo, nf, space)
        W = wake_operator(centers, pieces, interval, space, block=32, max_pairs=1000000)
        mu, gamma, residual = solve_augmented(body, rhs, W, te)
        g = basis_at(grid, space) @ gamma
        out["levels"].append(dict(factor=factor, dofs=int(space.hat.shape[0]),
                                  gamma=g.tolist(),
                                  gamma_sum=float(np.sum(g))))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps({"wall_panels": nf, "levels": [l["dofs"] for l in out["levels"]]}))


if __name__ == "__main__":
    main()