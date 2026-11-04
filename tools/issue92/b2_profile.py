"""Profile one coupled iteration on the CAD case with the torch CPU backend."""
import cProfile
import io
import pstats
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from panel.p1_coupled_material_wake import solve_coupled_material_wake
from tools.issue92.s5_loop_parity import _mesh

mesh = _mesh("g2_cad")
profile = cProfile.Profile()
profile.enable()
out = solve_coupled_material_wake(
    mesh, 5., iterations=1, kutta="pressure", pressure_solver="structured",
    solve_backend="torch", solve_device="cpu", analytic_backend="torch",
    analytic_device="cpu", gate_mode="audit")
profile.disable()
print("status", out["status"])
stream = io.StringIO()
pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(28)
print(stream.getvalue()[:6000])